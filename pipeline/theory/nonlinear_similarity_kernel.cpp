// Original nonlinear HJB in z=y/sqrt(terminal_variance+reverse_time),
// s=log(1+reverse_time/terminal_variance); no leading-order policy.
// Original-game nonlinear HJB, differentiated in state.
// Finite-precision output uses a positive generator; grid refinement is mandatory.
// Fully implicit BDF2 solution of the coupled nonlinear value HJB.
// No explicit transport CFL restriction. The observation law is unchanged.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

using M=std::array<double,4>;
using V=std::array<double,2>;
static M mul(M a,M b) {return {a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3]};}
static V mv(M a,V b) {return {a[0]*b[0]+a[1]*b[1],a[2]*b[0]+a[3]*b[1]};}
static M minus(M a,M b) {for(int i=0;i<4;++i)a[i]-=b[i];return a;}
static V minus(V a,V b) {return {a[0]-b[0],a[1]-b[1]};}
static M inv(M a) {double d=a[0]*a[3]-a[1]*a[2];return {a[3]/d,-a[1]/d,-a[2]/d,a[0]/d};}

extern "C" int solve_hjb_similarity(double wi,double wj,double h,double nu,double dx,int nt,
                         double extent,int max_work,double x0,double* outputs,int* work_steps) {
 int half=(int)std::ceil(extent/dx),n=2*half+1;
 double lap=.5/(dx*dx),maxw=std::max(wi,wj);
 std::vector<V> u(n),old(n),older(n),base(n),res(n),step(n),trial(n),slopes(n),flux(n-1);
 std::vector<M> upper(n),lower(n),diag(n),elim(n),upper2(n),lower2(n),elim2(n),fl(n-1),fr(n-1);
 std::vector<std::array<M,4>> face_jac(n-1);
 std::vector<V> slope_left(n),slope_right(n);
 auto normalized_variance=[&](double s){double steady=1/std::sqrt(nu),z=std::tanh(std::sqrt(nu)*s);return steady*(h+steady*z)/(steady+h*z);};
 double terminal_h=normalized_variance(1.0);
 double finish=std::log1p(1/terminal_h),dt=finish/nt;
 for(int j=0;j<n;++j){double y=(j-half)*dx;double g=std::exp(-y*y/2)/std::sqrt(2*std::acos(-1.0));u[j]={g,g};}
 u[0]={0,0};u[n-1]={0,0};old=u;older=u;*work_steps=0;
 auto minmod=[](double a,double b){return a*b<=0?0:std::copysign(std::min(std::abs(a),std::abs(b)),a);};
 auto residual=[&](const std::vector<V>& x,double alpha,double ds,bool jac){
   for(int j=1;j<n-1;++j)for(int q=0;q<2;++q){
    double dl=x[j][q]-x[j-1][q],dr=x[j+1][q]-x[j][q];slopes[j][q]=(nt<=64 && maxw>12)?0:minmod(dl,dr);
    slope_left[j][q]=!(nt<=64 && maxw>12) && dl*dr>0 && std::abs(dl)<=std::abs(dr)?1:0;
    slope_right[j][q]=!(nt<=64 && maxw>12) && dl*dr>0 && std::abs(dl)>std::abs(dr)?1:0;
   }
   for(int j=0;j<n-1;++j){
    V l={x[j][0]+slopes[j][0]/2,x[j][1]+slopes[j][1]/2},r={x[j+1][0]-slopes[j+1][0]/2,x[j+1][1]-slopes[j+1][1]/2};
    double face=(j+.5-half)*dx;
    double speed=2*maxw*std::max(l[0]+l[1],r[0]+r[1])+.5*std::abs(face);
    V lf={wj*l[0]*l[1]-wi*l[0]*l[0]/2,wj*l[1]*l[1]/2-wi*l[0]*l[1]};
    V rf={wj*r[0]*r[1]-wi*r[0]*r[0]/2,wj*r[1]*r[1]/2-wi*r[0]*r[1]};
    for(int q=0;q<2;++q){lf[q]-=.5*face*l[q];rf[q]-=.5*face*r[q];}
    flux[j]={(lf[0]+rf[0]-speed*(r[0]-l[0]))/2,(lf[1]+rf[1]-speed*(r[1]-l[1]))/2};
    if(jac){
      fl[j]={(-wi*l[0]+wj*l[1]+speed-.5*face)/2,wj*l[0]/2,-wi*l[1]/2,(wj*l[1]-wi*l[0]+speed-.5*face)/2};
      fr[j]={(-wi*r[0]+wj*r[1]-speed-.5*face)/2,wj*r[0]/2,-wi*r[1]/2,(wj*r[1]-wi*r[0]-speed-.5*face)/2};
      bool left=l[0]+l[1]>=r[0]+r[1];M& a=left?fl[j]:fr[j];
      for(int q=0;q<2;++q){double add=-maxw*(r[q]-l[q]);a[q*2]+=add;a[q*2+1]+=add;}
      V ld[4]={{-.5*slope_left[j][0],-.5*slope_left[j][1]},
               {1+.5*(slope_left[j][0]-slope_right[j][0]),1+.5*(slope_left[j][1]-slope_right[j][1])},
               {.5*slope_right[j][0],.5*slope_right[j][1]},{0,0}};
      V rd[4]={{0,0},{.5*slope_left[j+1][0],.5*slope_left[j+1][1]},
               {1-.5*(slope_left[j+1][0]-slope_right[j+1][0]),1-.5*(slope_left[j+1][1]-slope_right[j+1][1])},
               {-.5*slope_right[j+1][0],-.5*slope_right[j+1][1]}};
      for(int p=0;p<4;++p)for(int q=0;q<4;++q)face_jac[j][p][q]=fl[j][q]*ld[p][q%2]+fr[j][q]*rd[p][q%2];
    }
   }
   double norm=0;
   for(int j=1;j<n-1;++j){
     for(int q=0;q<2;++q){res[j][q]=alpha*x[j][q]-base[j][q]-ds*(lap*(x[j-1][q]-2*x[j][q]+x[j+1][q])-(flux[j][q]-flux[j-1][q])/dx);norm=std::max(norm,std::abs(res[j][q]));}
     if(jac){
       lower2[j]=lower[j]=diag[j]=upper[j]=upper2[j]=M{0,0,0,0};
       M* bands[5]={&lower2[j],&lower[j],&diag[j],&upper[j],&upper2[j]};
       for(int p=0;p<4;++p)for(int q=0;q<4;++q){
        (*bands[p+1])[q]+=ds/dx*face_jac[j][p][q];
        (*bands[p])[q]-=ds/dx*face_jac[j-1][p][q];
       }
       lower[j][0]-=ds*lap;lower[j][3]-=ds*lap;upper[j][0]-=ds*lap;upper[j][3]-=ds*lap;
       diag[j][0]+=alpha+2*ds*lap;diag[j][3]+=alpha+2*ds*lap;
     }
   }return norm;
 };
 std::vector<double> gain(n,0.),ga(n),gb(n),gc(n),rhs(n);
 auto output_step=[&](const std::vector<V>& before,const std::vector<V>& after,double now,double ds){
   double rho=terminal_h*std::exp(now+ds/2),physical=std::clamp(1-rho+terminal_h,0.0,1.0),hs=normalized_variance(physical),D=nu*hs*hs,Q=hs+1-physical;
   for(int j=1;j<n-1;++j){
     double z=(j-half)*dx,mi=wi*(before[j][0]+after[j][0])/2,mj=wj*(before[j][1]+after[j][1])/2,b=.5*z+mi-mj;
     double left=.5*D/(dx*dx)-b/(2*dx),right=.5*D/(dx*dx)+b/(2*dx);
     if(left<0 || right<0){left=.5*D/(dx*dx)+std::max(-b,0.0)/dx;right=.5*D/(dx*dx)+std::max(b,0.0)/dx;}
     double probability=.5*std::erfc(-z*std::sqrt(rho/(2*Q))),reward=mi*probability+mj*(1-probability);
     ga[j]=-ds*left/2;gc[j]=-ds*right/2;gb[j]=1+ds*(left+right+.5)/2;
     rhs[j]=gain[j]+ds/2*(left*gain[j-1]-(left+right+.5)*gain[j]+right*gain[j+1])+ds*reward;
   }
   for(int j=2;j<n-1;++j){double factor=ga[j]/gb[j-1];gb[j]-=factor*gc[j-1];rhs[j]-=factor*rhs[j-1];}
   gain[n-2]=rhs[n-2]/gb[n-2];for(int j=n-3;j>=1;--j)gain[j]=(rhs[j]-gc[j]*gain[j+1])/gb[j];
 };
 std::vector<double> targets;
 for(int k=1;k<=nt;++k)targets.push_back(finish*k/nt);
 double rate=nu*h,layer_end=std::min(1.0,30/rate),smallest=std::min(layer_end/5000,1.0/(nt*100));
 if(smallest>0 && layer_end>smallest)for(int k=0;k<240;++k){double physical=smallest*std::exp(std::log(layer_end/smallest)*k/239);double target=std::log1p(std::max(0.0,1-physical)/terminal_h);if(target>1e-11 && target<finish-1e-11)targets.push_back(target);}
 std::sort(targets.begin(),targets.end());targets.erase(std::unique(targets.begin(),targets.end()),targets.end());
 double tau=0,previous_dt=0,next_dt=std::min(dt,.02/std::max(1.0,maxw));int accepted_steps=0;
 for(double target:targets){
  if(target<=tau+1e-11)continue;
  while(tau<target-1e-11){
   double ds=std::min(target-tau,next_dt);if(previous_dt>0)ds=std::min(ds,1.5*previous_dt);
   bool accepted=false;
   for(int retry=0;retry<24;++retry){
    if(ds<1e-13 || ++*work_steps>max_work)return -6;
    double physical=std::clamp(1-terminal_h*std::expm1(tau+ds),0.0,1.0),v=normalized_variance(physical);lap=.5*nu*v*v/(dx*dx);
    bool startup=accepted_steps<2;double ratio=previous_dt>0?ds/previous_dt:1,alpha=startup?1:(1+2*ratio)/(1+ratio);
    for(int j=1;j<n-1;++j){for(int q=0;q<2;++q)base[j][q]=startup?old[j][q]:(1+ratio)*old[j][q]-ratio*ratio/(1+ratio)*older[j][q];u[j]=old[j];}
    bool converged=false;
    for(int it=0;it<24;++it){
     double err=residual(u,alpha,ds,true);
     if(!std::isfinite(err))break;
     if(err<1e-10){converged=true;break;}
     for(int j=1;j<n-1;++j)step[j]={-res[j][0],-res[j][1]};
     for(int j=1;j<n-1;++j){
       M inverse=inv(diag[j]);elim[j]=mul(inverse,upper[j]);elim2[j]=mul(inverse,upper2[j]);step[j]=mv(inverse,step[j]);
       if(j+1<n-1){diag[j+1]=minus(diag[j+1],mul(lower[j+1],elim[j]));upper[j+1]=minus(upper[j+1],mul(lower[j+1],elim2[j]));step[j+1]=minus(step[j+1],mv(lower[j+1],step[j]));}
       if(j+2<n-1){lower[j+2]=minus(lower[j+2],mul(lower2[j+2],elim[j]));diag[j+2]=minus(diag[j+2],mul(lower2[j+2],elim2[j]));step[j+2]=minus(step[j+2],mv(lower2[j+2],step[j]));}
     }
     for(int j=n-3;j>=1;--j){step[j]=minus(step[j],mv(elim[j],step[j+1]));if(j+2<n-1)step[j]=minus(step[j],mv(elim2[j],step[j+2]));}
     double factor=1;bool descended=false;
     for(int ls=0;ls<16;++ls){
      trial=u;bool positive=true;
      for(int j=1;j<n-1;++j)for(int q=0;q<2;++q){trial[j][q]=u[j][q]+factor*step[j][q];positive=positive&&trial[j][q]>=-1e-11;}
      double next=positive?residual(trial,alpha,ds,false):INFINITY;
      if(std::isfinite(next)&&next<err){u.swap(trial);descended=true;break;}factor*=.5;
     }
     if(!descended)break;
    }
    if(converged){accepted=true;break;}ds*=.5;
   }
   if(!accepted)return -5;
   output_step(old,u,tau,ds);older=old;old=u;previous_dt=ds;tau+=ds;++accepted_steps;next_dt=std::min(dt,ds*1.4);
  }
  tau=target;
 }
 double pos=x0/std::sqrt(1+terminal_h)/dx+half;
 if(pos<0 || pos>=n-1)return -7;int j=(int)pos;double f=pos-j;
 outputs[0]=std::sqrt(1+terminal_h)*((1-f)*gain[j]+f*gain[j+1]);outputs[1]=outputs[2]=0;
 for(int j=0;j<n;++j){if(!std::isfinite(u[j][0])||!std::isfinite(u[j][1])||u[j][0]<-1e-10||u[j][1]<-1e-10)return -1;outputs[1]+=u[j][0]*dx;outputs[2]+=u[j][1]*dx;}
 return n;
}

