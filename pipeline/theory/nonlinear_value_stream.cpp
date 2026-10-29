// Streaming implementation of joint_design_exact's adaptive CN/Heun scheme.
// Original normalized value HJB, float32 policies, and the same output PDE.
// No strategy expansion or altered time discretization enters this kernel.
#include <algorithm>
#include <cmath>
#include <vector>

using V = std::vector<double>;

static void gradient(const V& v, double dx, V& g) {
    const int n = v.size();
    for (int k=1;k<n-1;++k) g[k]=(v[k+1]-v[k-1])/(2*dx);
    g[0]=(-1.5*v[0]+2*v[1]-.5*v[2])/dx;
    g[n-1]=(.5*v[n-3]-2*v[n-2]+1.5*v[n-1])/dx;
}

static void factor(int n, double r, V& inverse) {
    inverse[0]=1/(1+2*r);
    for(int k=1;k<n;++k) inverse[k]=1/(1+2*r-r*r*inverse[k-1]);
}

static void solve(const V& rhs, V& v, const V& inverse, double r) {
    const int n=inverse.size();
    v[1]=rhs[0]*inverse[0];
    for(int k=1;k<n;++k) v[k+1]=(rhs[k]+r*v[k])*inverse[k];
    for(int k=n-2;k>=0;--k) v[k+1]+=r*inverse[k]*v[k+2];
}

extern "C" int original_value_stream(int nt, int ny, double dy, double outdy,
    double theta, double ci, double cj, const double* time, const double* y,
    const double* diffusion, const double* qmid, const double* terminal,
    double* gain, double* diagnostics) {
    if(nt<2 || ny<5 || dy<=0 || outdy<=0 || theta<=0 || ci<=0 || cj<=0) return -1;
    const int m=ny-2;
    V vi(ny),vj(ny),gi(ny),gj(ny),pi(ny),pj(ny),gpi(ny),gpj(ny);
    V oldi(ny),oldj(ny),newi(ny),newj(ny),drift(ny),reward(ny);
    V basei(m),basej(m),rhsi(m),rhsj(m),inverse(m),output(ny,0),pred(ny,0),base(m),rhs(m);
    for(int k=0;k<ny;++k){vi[k]=terminal[2*k];vj[k]=terminal[2*k+1];}
    double minpolicy=0,maxpolicy=0;
    auto policy=[&](V& mi,V& mj)->bool {
        gradient(vi,dy,gi);gradient(vj,dy,gj);
        for(int k=0;k<ny;++k) {
            const double a=gi[k]*theta/ci,b=gj[k]*(-theta/cj);
            if(!std::isfinite(a) || !std::isfinite(b)) return false;
            minpolicy=std::min(minpolicy,std::min(a,b));
            mi[k]=static_cast<float>(std::max(0.,a));
            mj[k]=static_cast<float>(std::max(0.,b));
            maxpolicy=std::max(maxpolicy,std::max(mi[k],mj[k]));
        }
        return minpolicy>=-1e-7;
    };
    if(!policy(oldi,oldj)){diagnostics[0]=minpolicy;return -2;}
    for(int index=nt-1;index>0;--index) {
        const double step=time[index]-time[index-1];
        const double r=step*diffusion[index-1]/(4*dy*dy);
        factor(m,r,inverse);
        pi=vi;pj=vj;
        for(int k=1;k<ny-1;++k) {
            basei[k-1]=vi[k]+r*(vi[k-1]-2*vi[k]+vi[k+1]);
            basej[k-1]=vj[k]+r*(vj[k-1]-2*vj[k]+vj[k+1]);
        }
        basei[0]+=r*vi[0];basei[m-1]+=r*vi[ny-1];
        basej[0]+=r*vj[0];basej[m-1]+=r*vj[ny-1];
        for(int k=1;k<ny-1;++k) {
            const double cross=gi[k]*gj[k];
            rhsi[k-1]=basei[k-1]+step*theta*(gi[k]*gi[k]/(2*ci)+cross/cj);
            rhsj[k-1]=basej[k-1]+step*theta*(gj[k]*gj[k]/(2*cj)+cross/ci);
        }
        solve(rhsi,pi,inverse,r);solve(rhsj,pj,inverse,r);
        gradient(pi,dy,gpi);gradient(pj,dy,gpj);
        for(int k=1;k<ny-1;++k) {
            const double cross=gi[k]*gj[k],nextcross=gpi[k]*gpj[k];
            rhsi[k-1]=basei[k-1]+step*theta*((gi[k]*gi[k]/(2*ci)+cross/cj)
                        +(gpi[k]*gpi[k]/(2*ci)+nextcross/cj))/2;
            rhsj[k-1]=basej[k-1]+step*theta*((gj[k]*gj[k]/(2*cj)+cross/ci)
                        +(gpj[k]*gpj[k]/(2*cj)+nextcross/ci))/2;
        }
        solve(rhsi,vi,inverse,r);solve(rhsj,vj,inverse,r);
        if(!policy(newi,newj)){diagnostics[0]=minpolicy;diagnostics[1]=index-1;return -2;}
        const double ro=step*diffusion[index-1]/(4*outdy*outdy);
        factor(m,ro,inverse);
        pred=output;
        for(int k=1;k<ny-1;++k) {
            const double mi=(oldi[k]+newi[k])/2,mj=(oldj[k]+newj[k])/2;
            drift[k]=mi-mj;
            const double prob=.5*std::erfc(-y[k]/std::sqrt(2*qmid[index-1]));
            reward[k]=(mi+mj)/2+drift[k]*(prob-.5);
            base[k-1]=output[k]+ro*(output[k-1]-2*output[k]+output[k+1]);
            const double flow=drift[k]*(output[k+1]-output[k-1])/(2*outdy)+reward[k];
            rhs[k-1]=base[k-1]+step*flow;
        }
        solve(rhs,pred,inverse,ro);
        for(int k=1;k<ny-1;++k) {
            const double first=drift[k]*(output[k+1]-output[k-1])/(2*outdy)+reward[k];
            const double second=drift[k]*(pred[k+1]-pred[k-1])/(2*outdy)+reward[k];
            rhs[k-1]=base[k-1]+step*(first+second)/2;
        }
        solve(rhs,output,inverse,ro);
        oldi.swap(newi);oldj.swap(newj);
    }
    double low=1.,high=0.;
    for(int k=0;k<ny;++k) {
        if(!std::isfinite(vi[k]) || !std::isfinite(vj[k]) || !std::isfinite(output[k])) return -3;
        low=std::min(low,std::min(vi[k],vj[k]));high=std::max(high,std::max(vi[k],vj[k]));
        gain[k]=output[k];
    }
    diagnostics[0]=minpolicy;diagnostics[1]=maxpolicy;diagnostics[2]=low;diagnostics[3]=high;
    return 1;
}
