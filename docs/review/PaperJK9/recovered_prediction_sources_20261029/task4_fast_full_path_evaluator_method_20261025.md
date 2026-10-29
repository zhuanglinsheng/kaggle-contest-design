# Task 4：一次前向递推求值完整预测概率路径

只扩展已核验v2审查求值器的输出：训练截止立即保存训练似然；其后每小时输出同一左端递推下的联合Poisson计数密度。一次路径递推替代每小时重复整个前缀，便于同目标独立重要性积分普通小时。没有重新估计参数或改变模型。编译产物和完整源码保留在独立审查目录，必须核对v2训练密度及独立Python逐时概率后才能用于汇总。数值失败后的路径保持负无穷；训练密度不受未来失败影响，生产积分须报告此类路径的训练后验权重。

```cpp
#include <cmath>
#include <algorithm>
#include <limits>
extern "C" void batch_forecast_path(int rows,int dim,const double* points,int trainN,int totalN,int countStart,double dt,double T,double theta,const double* held,const int* ciobs,const int* cjobs,double* trainOut,double* predictOut){
 const double neg=-std::numeric_limits<double>::infinity(), pi=std::acos(-1.0),logeps=std::log(1e-12);
 for(int row=0;row<rows;++row){
  trainOut[row]=neg;for(int j=0;j<totalN-trainN;++j)predictOut[row*(totalN-trainN)+j]=neg;const double* z=points+row*dim;
  bool invalid=false;for(int j=0;j<dim;++j)if(!std::isfinite(z[j]))invalid=true;
  if(invalid||dim<6)continue;
  double sigma=std::exp(z[2]),lam=std::exp(z[3]),sig2=sigma*sigma,bar=sigma/std::sqrt(lam),K=-std::expm1(-std::sqrt(lam)*sigma*dt),Qref=bar+sig2*T;
  if(!(bar>0)||!std::isfinite(Qref))continue;
  double y=z[4],drift=0.,resmean=0.,resvar=bar,ll=0.;int previous=0;
  for(int h=0;h<totalN;++h){
   double Q=bar+sig2*(T-h*dt);double logkernel;
   if(dim==8)logkernel=-.5*std::log(2*pi*Qref)-z[6]*std::log(Q/Qref)-std::exp(z[7])*y*y/(2*Q);
   else logkernel=-.5*std::log(2*pi*Q)-y*y/(2*Q);
   double lmi=std::log(theta)-z[0]+logkernel,lmj=std::log(theta)-z[1]+logkernel;
   double mi=std::exp(lmi),mj=std::exp(lmj),li=lmi+z[5]+std::log(dt),lj=lmj+z[5]+std::log(dt),vi=std::exp(li),vj=std::exp(lj);
   if(!std::isfinite(mi)||!std::isfinite(mj)||!std::isfinite(vi)||!std::isfinite(vj)){invalid=true;break;}
   if(h<trainN){
    drift+=(mi-mj)*dt;
    if(ciobs[h])ll+=ciobs[h]*std::max(li,logeps);
    if(cjobs[h])ll+=cjobs[h]*std::max(lj,logeps);
    if(h>=countStart)ll-=vi+vj;
    if(ciobs[h]+cjobs[h]>0){
     double gap=(h+1-previous)*dt,predvar=resvar+sig2*gap,measure=1/(lam*gap)+1e-9,V=predvar+measure,innovation=held[h]-z[4]-drift-resmean;
     ll+=-.5*(std::log(2*pi*V)+innovation*innovation/V);
     resmean+=predvar/V*innovation;resvar=predvar*measure/V;previous=h+1;
    }
   }
   if(h==trainN-1 && std::isfinite(ll))trainOut[row]=ll;
   if(h>=trainN){
    int testI=ciobs[h],testJ=cjobs[h];
    double lp=-vi-vj-std::lgamma(testI+1.)-std::lgamma(testJ+1.);
    if(testI)lp+=testI*li;if(testJ)lp+=testJ*lj;
    if(std::isfinite(ll)&&!std::isnan(lp))predictOut[row*(totalN-trainN)+h-trainN]=lp;
   }
   y=y+(mi-mj)*dt+K*(held[h]-y);
   if(!std::isfinite(y)||!std::isfinite(ll)){invalid=true;break;}
  }
 }
}

```
