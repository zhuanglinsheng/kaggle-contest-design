# Task 4：独立训练密度与单小时预测概率求值器

用途：尾部预测概率不能以极少后验贡献样本估得可靠，需要能够对已冻结训练目标和单小时预测密度分别积分。该审查求值器逐点复现同一训练似然（含事件log强度地板、初始化、左端努力、Gaussian创新与曝光），预测则使用规范化Poisson概率。C/D均支持，原六参数先验及C额外先验在外部log坐标加入。

预测时只沿已知公共路径递推至当前小时，不把此前或当前测试计数加入训练似然。当前观测计数仅作为被评分分布的求值点；后续若用其构造重要性提议，仍必须通过精确密度权重计算同一个固定后验预测积分，不能将其称作训练模型更新。

编译产物仅在审查目录；下面是通过stdin编译的完整C++源。使用前必须与原Stan完整密度及独立Python预测核对，编译成功本身不证明数值等价。

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
   if(h==predictH){
    double lp=-vi-vj-std::lgamma(testI+1.)-std::lgamma(testJ+1.);
    if(testI)lp+=testI*li;if(testJ)lp+=testJ*lj;
    if(std::isfinite(ll)&&!std::isnan(lp)){trainOut[row]=ll;predictOut[row]=lp;}
    break;
   }
   y=y+(mi-mj)*dt+K*(held[h]-y);
   if(!std::isfinite(y)||!std::isfinite(ll)){invalid=true;break;}
  }
 }
}

```
