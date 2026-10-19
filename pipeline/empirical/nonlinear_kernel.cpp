// Numerical steady-state HJB and the unchanged empirical observation likelihood.
// Dimensionless time t/T and state y/(sigma sqrt(T)) give diffusion one,
// incentive coefficients w_k=theta/(sigma^2 c_k), and terminal variance h.
#include <algorithm>
#include <cmath>
#include <vector>

static void gradient(const std::vector<double>& u, std::vector<double>& g, double dx) {
  const int n=u.size();
  for(int j=1;j<n-1;++j) g[j]=(u[j+1]-u[j-1])/(2*dx);
  g[0]=(-3*u[0]+4*u[1]-u[2])/(2*dx);
  g[n-1]=(3*u[n-1]-4*u[n-2]+u[n-3])/(2*dx);
}

static void tridiagonal(std::vector<double>& v,double r) {
  const int n=v.size();
  std::vector<double> c(n);
  double denom=1+2*r;
  c[0]=-r/denom; v[0]/=denom;
  for(int j=1;j<n;++j) {
    denom=1+2*r+r*c[j-1];
    c[j]=(j==n-1?0:-r/denom);
    v[j]=(v[j]+r*v[j-1])/denom;
  }
  for(int j=n-2;j>=0;--j) v[j]-=c[j]*v[j+1];
}

extern "C" int solve_hjb(double wi,double wj,double h,double dx,int nt,
                         double extent,double* policies,int* work_steps) {
  int half=(int)std::ceil(extent*std::sqrt(1+h)/dx), n=2*half+1;
  // Differentiate the HJB: a=U_i,y and b=-U_j,y are nonnegative densities.
  // Their conservative fluxes are (wj*a*b-wi*a*a/2, wj*b*b/2-wi*a*b).
  // MUSCL reconstruction, local Rusanov flux, and SSP Heun preserve positive
  // policies without the global first-order artificial diffusion of value solvers.
  std::vector<double> a(n),b(n),sa(n),sb(n),fa(n-1),fb(n-1),
    da(n),db(n),pa(n),pb(n),qa(n),qb(n),rhs(n-2);
  for(int j=0;j<n;++j) {
    double y=(j-half)*dx;
    a[j]=b[j]=std::exp(-y*y/(2*h))/std::sqrt(2*std::acos(-1.0)*h);
  }
  a[0]=a[n-1]=b[0]=b[n-1]=0;
  auto minmod=[](double x,double y) {
    if(x*y<=0) return 0.0;
    return std::copysign(std::min(std::abs(x),std::abs(y)),x);
  };
  auto diffuse=[&](std::vector<double>& v,double duration) {
    double r=duration/(4*dx*dx);
    for(int j=1;j<n-1;++j) rhs[j-1]=v[j]+r*(v[j-1]-2*v[j]+v[j+1]);
    tridiagonal(rhs,r);
    for(int j=1;j<n-1;++j) v[j]=rhs[j-1];
  };
  auto flow=[&](const std::vector<double>& aa,const std::vector<double>& bb) {
    sa[0]=sa[n-1]=sb[0]=sb[n-1]=0;
    for(int j=1;j<n-1;++j) {
      sa[j]=minmod(aa[j]-aa[j-1],aa[j+1]-aa[j]);
      sb[j]=minmod(bb[j]-bb[j-1],bb[j+1]-bb[j]);
    }
    for(int j=0;j<n-1;++j) {
      double al=aa[j]+sa[j]/2,ar=aa[j+1]-sa[j+1]/2;
      double bl=bb[j]+sb[j]/2,br=bb[j+1]-sb[j+1]/2;
      double speed=2*std::max(wi,wj)*std::max(al+bl,ar+br);
      fa[j]=(wj*al*bl-wi*al*al/2+wj*ar*br-wi*ar*ar/2-speed*(ar-al))/2;
      fb[j]=(wj*bl*bl/2-wi*al*bl+wj*br*br/2-wi*ar*br-speed*(br-bl))/2;
    }
    da[0]=da[n-1]=db[0]=db[n-1]=0;
    for(int j=1;j<n-1;++j) {
      da[j]=-(fa[j]-fa[j-1])/dx;
      db[j]=-(fb[j]-fb[j-1])/dx;
    }
  };
  double time=1;
  *work_steps=0;
  for(int k=nt;k>=0;--k) {
    for(int j=0;j<n;++j) {
      if(!std::isfinite(a[j]) || !std::isfinite(b[j]) || a[j]<-1e-10 || b[j]<-1e-10) return -1;
      policies[((size_t)k*n+j)*2]=wi*std::max(0.0,a[j]);
      policies[((size_t)k*n+j)*2+1]=wj*std::max(0.0,b[j]);
    }
    if(k==0) break;
    double target=(double)(k-1)/nt;
    while(time>target+1e-14) {
      double speed=0;
      for(int j=0;j<n;++j) {
        if(!std::isfinite(a[j]) || !std::isfinite(b[j]) || a[j]<-1e-10 || b[j]<-1e-10) return -1;
        speed=std::max(speed,2*std::max(wi,wj)*(a[j]+b[j]));
      }
      if(!std::isfinite(speed)) return -1;
      double dt=std::min(time-target,std::min(2*dx*dx,.15*dx/std::max(speed,1e-12)));
      if(++*work_steps>2000000) return -2;
      diffuse(a,dt/2); diffuse(b,dt/2);
      flow(a,b);
      for(int j=1;j<n-1;++j) {
        pa[j]=a[j]+dt*da[j];pb[j]=b[j]+dt*db[j];
      }
      flow(pa,pb);
      for(int j=1;j<n-1;++j) {
        a[j]=(a[j]+pa[j]+dt*da[j])/2;
        b[j]=(b[j]+pb[j]+dt*db[j])/2;
      }
      diffuse(a,dt/2);diffuse(b,dt/2);
      time-=dt;
    }
    time=target;
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
