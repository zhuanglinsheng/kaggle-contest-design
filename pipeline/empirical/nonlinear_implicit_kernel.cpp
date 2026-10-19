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

extern "C" int solve_hjb(double wi,double wj,double h,double dx,int nt,
                         double extent,double* policies,int* work_steps) {
 int half=(int)std::ceil(extent*std::sqrt(1+h)/dx),n=2*half+1;
 double dt=1.0/nt,lap=.5/(dx*dx),maxw=std::max(wi,wj);
 std::vector<V> u(n),old(n),older(n),base(n),res(n),step(n),trial(n),slopes(n),flux(n-1);
 std::vector<M> upper(n),lower(n),diag(n),elim(n),upper2(n),lower2(n),elim2(n),fl(n-1),fr(n-1);
 std::vector<std::array<M,4>> face_jac(n-1);
 std::vector<V> slope_left(n),slope_right(n);
 for(int j=0;j<n;++j){double y=(j-half)*dx;double g=std::exp(-y*y/(2*h))/std::sqrt(2*std::acos(-1.0)*h);u[j]={g,g};}
 u[0]={0,0};u[n-1]={0,0};old=u;older=u;*work_steps=0;
 auto minmod=[](double a,double b){return a*b<=0?0:std::copysign(std::min(std::abs(a),std::abs(b)),a);};
 auto residual=[&](const std::vector<V>& x,double alpha,double ds,bool jac){
   for(int j=1;j<n-1;++j)for(int q=0;q<2;++q){
    double dl=x[j][q]-x[j-1][q],dr=x[j+1][q]-x[j][q];slopes[j][q]=minmod(dl,dr);
    slope_left[j][q]=dl*dr>0 && std::abs(dl)<=std::abs(dr)?1:0;
    slope_right[j][q]=dl*dr>0 && std::abs(dl)>std::abs(dr)?1:0;
   }
   for(int j=0;j<n-1;++j){
    V l={x[j][0]+slopes[j][0]/2,x[j][1]+slopes[j][1]/2},r={x[j+1][0]-slopes[j+1][0]/2,x[j+1][1]-slopes[j+1][1]/2};
    double speed=2*maxw*std::max(l[0]+l[1],r[0]+r[1]);
    V lf={wj*l[0]*l[1]-wi*l[0]*l[0]/2,wj*l[1]*l[1]/2-wi*l[0]*l[1]};
    V rf={wj*r[0]*r[1]-wi*r[0]*r[0]/2,wj*r[1]*r[1]/2-wi*r[0]*r[1]};
    flux[j]={(lf[0]+rf[0]-speed*(r[0]-l[0]))/2,(lf[1]+rf[1]-speed*(r[1]-l[1]))/2};
    if(jac){
      fl[j]={(-wi*l[0]+wj*l[1]+speed)/2,wj*l[0]/2,-wi*l[1]/2,(wj*l[1]-wi*l[0]+speed)/2};
      fr[j]={(-wi*r[0]+wj*r[1]-speed)/2,wj*r[0]/2,-wi*r[1]/2,(wj*r[1]-wi*r[0]-speed)/2};
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
 auto store=[&](int k){for(int j=0;j<n;++j){if(!std::isfinite(u[j][0])||!std::isfinite(u[j][1])||u[j][0]<-1e-10||u[j][1]<-1e-10)return false;policies[((size_t)k*n+j)*2]=wi*std::max(0.0,u[j][0]);policies[((size_t)k*n+j)*2+1]=wj*std::max(0.0,u[j][1]);}return true;};
 if(!store(nt))return -1;
 double tau=0,previous_dt=0,next_dt=std::min(dt,.02*h/std::max(1.0,maxw));int accepted_steps=0;
 for(int k=nt-1;k>=0;--k){
  double target=(double)(nt-k)/nt;
  while(tau<target-1e-14){
   double ds=std::min(target-tau,next_dt);if(previous_dt>0)ds=std::min(ds,1.5*previous_dt);
   bool accepted=false;
   for(int retry=0;retry<24;++retry){
    if(ds<1e-13 || ++*work_steps>1000000)return -6;
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
   older=old;old=u;previous_dt=ds;tau+=ds;++accepted_steps;next_dt=std::min(dt,ds*1.4);
  }
  tau=target;if(!store(k))return -1;
 }
 return n;
}

extern "C" double empirical_likelihood(int n,double delta,double horizon,double theta,
  double ci,double cj,double sigma,double lambda,double mu,double log_r,
  const double* held,const int* counts_i,const int* counts_j,int count_start,
  const double* policy,int ny,int nt,double dx,int* status) {
  double steady=sigma/std::sqrt(lambda),scale=sigma*std::sqrt(horizon),
    effort_scale=sigma/std::sqrt(horizon),weight=-std::expm1(-std::sqrt(lambda)*sigma*delta),
    r=std::exp(log_r),perceived=mu,drift=0,residual=0,resvar=steady,ll=0;
  int previous_obs=0; *status=0;
  const double pi=std::acos(-1.0);
  for(int i=0;i<n;++i) {
    double mi,mj;
    if(policy) {
      double pos=perceived/scale/dx+(ny-1)/2.0;
      if(pos<1 || pos>ny-2) { *status=-3; return -INFINITY; }
      int j=(int)pos; double z=pos-j;
      double tp=(i*delta/horizon)*nt;
      int k=std::min((int)tp,nt-1); double tf=tp-k;
      auto interp=[&](int kk,int player){
        return (1-z)*policy[((size_t)kk*ny+j)*2+player]
          +z*policy[((size_t)kk*ny+j+1)*2+player];
      };
      mi=effort_scale*((1-tf)*interp(k,0)+tf*interp(k+1,0));
      mj=effort_scale*((1-tf)*interp(k,1)+tf*interp(k+1,1));
    } else {
      double q=steady+sigma*sigma*(horizon-i*delta);
      double kernel=std::exp(-perceived*perceived/(2*q))/std::sqrt(2*pi*q);
      mi=theta/ci*kernel; mj=theta/cj*kernel;
    }
    double ii=r*mi*delta,jj=r*mj*delta;
    ll+=counts_i[i]*std::log(std::max(ii,1e-12))+counts_j[i]*std::log(std::max(jj,1e-12));
    if(i>=count_start) ll-=ii+jj;
    drift+=(mi-mj)*delta;
    if(counts_i[i]+counts_j[i]>0) {
      double elapsed=(i+1-previous_obs)*delta;
      double predicted=resvar+sigma*sigma*elapsed;
      double noise=1/(lambda*elapsed)+1e-9,v=predicted+noise;
      double innovation=held[i]-mu-drift-residual;
      ll+=-.5*(std::log(2*pi*v)+innovation*innovation/v);
      residual+=predicted/v*innovation; resvar=predicted*noise/v;
      previous_obs=i+1;
    }
    perceived+=(mi-mj)*delta+weight*(held[i]-perceived);
  }
  if(!std::isfinite(ll)) *status=-4;
  return ll;
}
