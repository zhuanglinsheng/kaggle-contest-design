# Task 4：训练分母与未来数值路径分离

独立求值器v2在训练截止立即保存训练似然，之后任何未来递推的浮点失败仅影响当前预测求值，不改变训练后验分母。正常参数点的结果与v1相同；这是保证重要性积分分母只依赖训练数据的实现不变量。v1保留供追溯；全样本先验MH仅在完整训练末端求值，其目标不受此变化影响。

```cpp
#include <cmath>
#include <algorithm>
#include <limits>
extern "C" void batch_components(int rows,int dim,const double* points,int trainN,int predictH,int countStart,double dt,double T,double theta,const double* held,const int* ciobs,const int* cjobs,int testI,int testJ,double* trainOut,double* predictOut){
 const double neg=-std::numeric_limits<double>::infinity(), pi=std::acos(-1.0),logeps=std::log(1e-12);
 for(int row=0;row<rows;++row){
  trainOut[row]=neg;predictOut[row]=neg;const double* z=points+row*dim;
  bool invalid=false;for(int j=0;j<dim;++j)if(!std::isfinite(z[j]))invalid=true;
  if(invalid||dim<6)continue;
  double sigma=std::exp(z[2]),lam=std::exp(z[3]),sig2=sigma*sigma,bar=sigma/std::sqrt(lam),K=-std::expm1(-std::sqrt(lam)*sigma*dt),Qref=bar+sig2*T;
  if(!(bar>0)||!std::isfinite(Qref))continue;
  double y=z[4],drift=0.,resmean=0.,resvar=bar,ll=0.;int previous=0;
  for(int h=0;h<=predictH;++h){
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
   if(h==predictH){
    double lp=-vi-vj-std::lgamma(testI+1.)-std::lgamma(testJ+1.);
    if(testI)lp+=testI*li;if(testJ)lp+=testJ*lj;
    if(std::isfinite(ll)&&!std::isnan(lp))predictOut[row]=lp;
    break;
   }
   y=y+(mi-mj)*dt+K*(held[h]-y);
   if(!std::isfinite(y)||!std::isfinite(ll)){invalid=true;break;}
  }
 }
}

```
