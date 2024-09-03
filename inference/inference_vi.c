// MCMC algorithm for sampling from the posterior distribution of the iPAR model
// Stephen Catterall
// Assumes that the background infection rate depends ONLY on the eps parameters and the kernel parameter lambda
// Assumes that Number_Eps is (at least) 6
// NB To run on Linux, rename parameter file as parametersw.txt, then use tr -d '\r' <parametersw.txt> parameters.txt 

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h> 
#include <gsl/gsl_types.h>
#include <gsl/gsl_errno.h>

#define PI 3.1415926536
#define MNCP 5 // maximum number of change points    
#define MNU 5000 // maximum number of units/cells in modelled landscape
#define DMAX 100 //this is the maximum distance over which secondary transmission can occur (excluding potential background infection)
#define NBMAX 5000 // max number of units in a neighbourhood (usually this should be bounded by (2*DMAX+1)^2)
#define MNCOV 6 // max number of covariates
#define SAVELATENT 10 // save latent variables every SAVELATENT iterations
#define SAVEPROPSD 0
#define ALP 1.0 // parameter in Dirichlet prior
#define INCLUDETIMES 0  


struct unitss{
  int    x;
  int  	 y;
  double h[MNCOV]; // simplex covariates
  double k[MNCOV]; // exponential covariates
  int    timeclass; // index of the time sub-interval into which the unit's infection time falls
  int    initial; // initially infected
  int    new; // newly infected during the observed time interval
  double infection_time; // time at which unit becomes infected
  double old_infection_time; // for storing old infection time value
  double hazard;// Hazard function at i, H(i), is the rate on unit[i] at the instant it is infected; H(i)=unit[i].hazard*unit[i].suitability
  double old_hazard; // for storing old value of unit[i].hazard
  double survival;// Survival function at i, S(i), is the probability that the infection time of i exceeds unit[i].infection_time; S(i)=exp(-unit[i].survival*unit[i].suitability)
  double old_survival; // for storing old value of unit[i].survival
  double susceptibility;
  double old_susceptibility;
  double infectivity;
  double old_infectivity;
  double likelihood_contribution;
  double old_likelihood_contribution;
  double tp; // proposal sd for the unit's infection time (each unit has a distinct proposal sd)
  double ep[MNCOV]; // epsilon covariates: pest,plat,plit,da,db,dc or whatever
  double eps; // current value of (background infection rate)/suitability for this unit
  double eps_old; // old value of (background infection rate)/suitability
  double land; // proportion of land (for use in computation of logistic suitability
};
typedef struct unitss units;

//model parameters
struct parameterr{
  double current;
  double old;
  double propsd;
  double logprior;
  double oldlogprior;
  double norm; // normalisation constant (useful for the kernel parameter lambda)
  int prior_e;
  double prior_r;
  double prior_l;
  double prior_u;
};
typedef struct parameterr parameter;

struct vparameterr{
  double current[MNCOV];
  double old[MNCOV];
  double propsd; // 1/alpha
  double logprior;
  double oldlogprior;
};
typedef struct vparameterr vparameter;

int    	thin2;
int 	kk,seed,burninflag;

const gsl_rng_type *T;
gsl_rng *r;
FILE *fptr;
FILE *lfptr; // for the latent variables
FILE *pfptr;
FILE *iptr;
FILE *cfptr;
char fname[100];
char input[100];
char line[10000];

units unit[MNU];
int gobs[MNU];
int iobs[MNU];
int     nunits;
int     numobs;
int     nsubints; // number of sub-intervals into which the observed time interval is split, including the initial time sub-interval [0,0]
double  infection_lower_bound[1000];
double  infection_upper_bound[1000];
double subinttimes[1000];
double tk[1000];
int    PRINT2SCREEN=1;//if 1, print running commentary to screen; if 0, don't
int    thin2screen=100;//if printing to screen, this sets the number of iterations between prints (eg if equal to 10, outputs every 10th iteration to screen)

vparameter sigma; // susc parameters (in the simplex)
vparameter gama; // inf parameters (in the simplex)
parameter eps[MNCOV]; // parameters contributing to background infection rate
parameter sigmap[MNCOV]; // exponential parameters for susceptibility
parameter gamap[MNCOV]; // exponential parameters for infectivity
parameter rho; // overall transmission rate
parameter lambda; // from the power law kernel
parameter f[MNCP]; // f values to allow for time varying transmission
parameter xi[MNCOV-1]; // xi parameter =ilr(sigma)
parameter xi2[MNCOV-1]; // xi2 parameter =ilr(gamma)

double    kernel_matrix[DMAX+1][DMAX+1]={{0.0}};// kernel_matrix[x][y] stores baseline secondary transmission rate at relative coordinates [x][y] from source of infection
double    old_kernel_matrix[DMAX+1][DMAX+1]={{0.0}};
double timev[5];
int 	  nb[MNU][NBMAX]={{0}}; // list of N(i)\I(i) for each unit i
int 	  nbinfec[MNU][NBMAX]={{0}}; // list of N(i)^I(i) for each unit i
int 	  infec[MNU][NBMAX]={{0}}; // list of I(i)\N(i) for each unit i
int 	  nbsize[MNU]={0}; // size of nb for each unit (should always be <=NBMAX)
int 	  nbinfecsize[MNU]={0}; // size of nbinfec
int 	  infecsize[MNU]={0}; // size of infec
double    linknb[MNU][NBMAX]={{0.0}}; // matrix of link strengths between interacting units (ignoring susceptibility and infectivity factors), with entries mirroring those in matrix nb
double    linknbinfec[MNU][NBMAX]={{0.0}}; // NB linknbinfec[i][j]:= distance dependent transmission rate from nbinfec[i][j] to i
double    revlinknbinfec[MNU][NBMAX]={{0.0}}; // NB revlinknbinfec[i][j]:= distance dependent transmission rate from i to nbinfec[i][j]
double    linkinfec[MNU][NBMAX]={{0.0}}; // linkinfec[i][j]:= distance dependent transmission rate from i to infec[i][j]
//double    old_link[MNU][NBMAX]={{0.0}}; // not using this method - just recomputing the links via the old kernel matrix seems more efficient
double    time_propsd,mean_time_propsd;//controls change to infection times

int    i,ii,j,k,count,feasible,isbetaexp,islambdaexp,ise1exp,iseaexp,dmax,offset,z,numtk,model,repno,Number_Simp,Number_Eps,Number_Exp,burnin,mcmc,newcols;
double beta_min,beta_max,beta_rate,tau_max,alpha_max,lambda_min,lambda_max,lambda_rate,ea_min,ea_max,ea_rate,e1_min,e1_max,e1_rate,loglik,logprior,old_logprior,delta,exponent;
double ttemp;
double f1=1.02; // NEW! f1,f2 adaptively control the size of the propsds during the burnin period NB f1=f2=1 corresponds to the original non-adaptive code
double f2=0.99;
int    x_distance,y_distance;
int    iteration,iteration2;
int    possible;
double a,b;
double verysmall;
double prob, pacc;
double logpost, old_logpost;
double t_end;
int    no_timechanges;
int    no_paramchanges;
int    potential_timechanges;
int    potential_paramchanges;
double verysmalllogprior=-1000.0;
double veryverysmalllogprior=-2000.0;
double (*epscalc) (int); // function pointer for the epscalc function

double kernel(int,int);
double priorsimp(double *);
double priorparam(parameter *, double);
void   samplesigma(gsl_rng *,double *);
void   samplegama(gsl_rng *,double *);
void   samplesigmap(int,gsl_rng *,double *);
void   samplegamap(int,gsl_rng *,double *);
void   samplerho(gsl_rng *,double *);
void   samplelambda(gsl_rng *,double *);
void   sampleeps(int, gsl_rng *,double *);
void   sampletime(int,gsl_rng *,double *);
double rn(gsl_rng *);
double epscalcfull(int);
double epscalcmini(int);
double intpf(double, double,double *);
double intf(double, double,double *);
double evalf(double,double *);
double evalpf(double,double *);
void samplef(int, gsl_rng *,double *);
void ilr();
void ilr2();
void display_help(void);
void get_arguments(int,char **,int *);
void read_parameters(void);
void readint(int *,char *,FILE *);
void readparam(parameter *,char *,FILE *);
void vreadparam(vparameter *,char *,FILE *);


double epscalcfull(int i) // compute background infection rate for unit i (full model)
{
	double out;
	out=unit[i].ep[0]*eps[0].current+unit[i].ep[1]*eps[1].current+unit[i].ep[2]*eps[2].current+pow(unit[i].ep[3],-2.0*lambda.current)*eps[3].current+pow(unit[i].ep[4],-2.0*lambda.current)*eps[4].current+pow(unit[i].ep[5],-2.0*lambda.current)*eps[5].current;
	return(out);
}

double epscalcmini(int i) // compute background infection rate for unit i (minimal model)
{
	double out,min;

	min=unit[i].ep[5];
	if(unit[i].ep[3]<min) min=unit[i].ep[3];
	if(unit[i].ep[4]<min) min=unit[i].ep[4];
	out=(unit[i].ep[0]+unit[i].ep[1]+unit[i].ep[2])*eps[0].current+pow(min,-2.0*lambda.current)*eps[3].current;
	return(out);
}

int main(int argc, char *argv[]) 
{
// Initialise GSL random number generator and miscellaneous other things
delta=1.0; // a fixed value, used to assist in the sampling of simplex parameters
gsl_rng_env_setup();
T=gsl_rng_default;
r = gsl_rng_alloc(T);
seed=(int)(time(NULL)); // Set random seed using computer clock
gsl_rng_set(r,seed);
 
// Get arguments 
if(argc<=1)
  {
    display_help();
    exit(0);
  }
if(argc>1) get_arguments(argc,argv,&repno); // get arguments
if((repno<1)||(repno>5))
{
	printf("\nAre you sure repno is correct? you've chosen %d.",repno);
	exit(0);
}
// Set up output files
sprintf(fname,"par_%d.txt",repno);
fptr=fopen(fname,"w");
sprintf(fname,"parlatent_%d.txt",repno);
lfptr=fopen(fname,"w");
sprintf(fname,"propsd_%d.txt",repno);
if(SAVEPROPSD) pfptr=fopen(fname,"w");

// Read parameter file
read_parameters();
if(nunits>MNU) {printf("\nError!!! MNU<nunits..."); exit(1);}
if(numtk>MNCP) {printf("\nError!!! Inconsistency in the number of check points..."); exit(1);}

count=0;
for(i=0;i<1000;i++) {infection_lower_bound[i]=0; infection_upper_bound[i]=0;}
for(i=0;i<nunits;i++) {nbsize[i]=0; nbinfecsize[i]=0; infecsize[i]=0;} // very important this bit!
if(model==0) epscalc=&epscalcmini;
if(model==1) epscalc=&epscalcfull;

verysmall=exp(-100.0);
time_propsd=timev[repno-1];

t_end=0.0; // initalise t_end which will be the time at the end of the inference period
for(k=0;k<nsubints;k++)
  {
    if(k>0)infection_lower_bound[k]=subinttimes[k-1];
    if(k>0)infection_upper_bound[k]=subinttimes[k];
    if(subinttimes[k]>t_end)t_end=subinttimes[k];
  }


//initialise the units, these are the default values unless changed by inputed data
for(i=0;i<nunits;i++)
  {
	unit[i].x=0;
	unit[i].y=0;
	unit[i].timeclass=nsubints+1;
	for(k=0;k<Number_Simp;++k)unit[i].h[k]=0.0;
	for(k=0;k<Number_Exp;k++) unit[i].k[k]=0;
	unit[i].initial=0;
	unit[i].new=0;
	unit[i].hazard=1.0;
	unit[i].old_hazard=1.0;
	unit[i].survival=0.0; 
	unit[i].old_survival=0.0;
	unit[i].likelihood_contribution=0.0;
	unit[i].old_likelihood_contribution=0.0;
	unit[i].tp=time_propsd;
	unit[i].susceptibility=0.0;
	unit[i].old_susceptibility=0.0;
	unit[i].infectivity=0.0;
	unit[i].old_infectivity=0.0;
	unit[i].infection_time=0;
	unit[i].old_infection_time=0;
	unit[i].eps=0;
	unit[i].eps_old=unit[i].eps;
	for(k=0;k<Number_Eps;k++) unit[i].ep[k]=0;
	unit[i].land=0;
  }

// Read in covariates from file

cfptr=fopen("covariates.dat","r");

for(k=0;k<nunits;k++) // unit ordering is identical to (and defined by) that given in the covariates file(s)
  {
	fscanf(cfptr,"%d %d",&unit[k].x,&unit[k].y);
	for(kk=0;kk<Number_Simp;kk++) fscanf(cfptr,"%lf",&unit[k].h[kk]);
	for(kk=0;kk<Number_Eps;kk++) fscanf(cfptr,"%lf",&unit[k].ep[kk]);
	for(kk=0;kk<Number_Exp;kk++) fscanf(cfptr,"%lf",&unit[k].k[kk]);
	unit[k].eps=epscalc(k);
  }
fclose(cfptr);

cfptr=fopen("data.dat","r");
newcols=0;
fscanf(cfptr,"%d %d",&numobs,&k);
for(k=0;k<numobs;++k)
  {
    fscanf(cfptr,"%d %d",&i,&j);
	iobs[k]=i;
	gobs[k]=j;
	unit[i].timeclass=j;
    if(j==0) {unit[i].initial=1; printf("I"); unit[i].infection_time=0;} // set infection time of an 'initial' to be zero (simplifies computation of likelihood later)
    if(j!=0)
	{
	  unit[i].new=1;
	  printf("N");
	  a=infection_lower_bound[j];
	  b=infection_upper_bound[j];
	  unit[i].infection_time=a+(b-a)*gsl_rng_uniform(r);
	  newcols++;
	}
  }
fclose(cfptr);

// Define neighbourhoods i.e. N(i)\I(i), N(i)^I(i), I(i)\N(i)
// For the current model, kernel is symmetric and purely distance-based, so N(i)=I(i) and only N(i)^I(i) is non-empty.
// Still, the code allows for the other two 'components' to be non-empty
for(i=0;i<nunits;i++)
{
	// Compute nbhd for unit i
	for(j=0;j<nunits;j++)
	{
	  if(kernel(unit[i].x-unit[j].x,unit[i].y-unit[j].y)>0) {nbinfec[i][nbinfecsize[i]]=j; nbinfecsize[i]++;} // NB assumes that lambda.current is sensible i.e. >0
	}
	  printf("\nunit %d: sizes are %d %d %d...",i,nbsize[i],nbinfecsize[i],infecsize[i]);
}
  printf("\nCheck dmax=%d but dmaxmax=%d",dmax,DMAX);

  //do the initialisation to work out initial likelihood

b=0.0;
for(x_distance=0;x_distance<=dmax;++x_distance)
  {
    for(y_distance=0;y_distance<=dmax;++y_distance)
	{
	  kernel_matrix[x_distance][y_distance]=kernel(x_distance,y_distance);
	  if((x_distance==0)&&(y_distance!=0))b=b+2.0*kernel_matrix[x_distance][y_distance];
	  if((x_distance!=0)&&(y_distance==0))b=b+2.0*kernel_matrix[x_distance][y_distance];
	  if((x_distance!=0)&&(y_distance!=0))b=b+4.0*kernel_matrix[x_distance][y_distance];
	}
  }
for(x_distance=0;x_distance<=dmax;++x_distance){for(y_distance=0;y_distance<=dmax;++y_distance)kernel_matrix[x_distance][y_distance]=kernel_matrix[x_distance][y_distance]/b;}//normalises the kernel_matrix

// Compute link strength for each interacting pair of units AND compute susceptibility/infectivity for each unit
//sigma=0;
for(i=0;i<nunits;i++)
{
	for(j=0;j<nbsize[i];j++) linknb[i][j]=kernel_matrix[abs(unit[i].x-unit[nb[i][j]].x)][abs(unit[i].y-unit[nb[i][j]].y)];
	for(j=0;j<nbinfecsize[i];j++) {linknbinfec[i][j]=kernel_matrix[abs(unit[i].x-unit[nbinfec[i][j]].x)][abs(unit[i].y-unit[nbinfec[i][j]].y)]; revlinknbinfec[i][j]=linknbinfec[i][j];}
	for(j=0;j<infecsize[i];j++) linkinfec[i][j]=kernel_matrix[abs(unit[i].x-unit[infec[i][j]].x)][abs(unit[i].y-unit[infec[i][j]].y)];
	unit[i].susceptibility=0;
	for(k=0;k<Number_Simp;++k) unit[i].susceptibility += sigma.current[k]*unit[i].h[k];
	exponent=0;
	for(k=0;k<Number_Exp;k++) exponent+=sigmap[k].current*unit[i].k[k];
	unit[i].susceptibility*=exp(exponent);
	unit[i].infectivity=0;
	for(k=0;k<Number_Simp;++k) unit[i].infectivity += gama.current[k]*unit[i].h[k];
	exponent=0;
	for(k=0;k<Number_Exp;k++) exponent+=gamap[k].current*unit[i].k[k];
	unit[i].infectivity*=exp(exponent);
}

//for(i=0;i<nunits;i++) unit[i].suitability /= sigma;

  feasible=0; // look for a feasible initial infection time sequence i.e. one whose probability exceeds zero
  if(lambda.propsd>verysmall) {lambda.logprior=priorparam(&lambda,lambda.current);} else {lambda.logprior=0;}
  if(rho.propsd>verysmall) {rho.logprior=priorparam(&rho,rho.current);} else {rho.logprior=0;}
  for(k=0;k<Number_Eps;++k){if(eps[k].propsd>verysmall) {eps[k].logprior=priorparam(&eps[k],eps[k].current);} else {eps[k].logprior=0;}}
  if(sigma.propsd>verysmall) {sigma.logprior=priorsimp(sigma.current);} else {sigma.logprior=0;}
  if(gama.propsd>verysmall) {gama.logprior=priorsimp(gama.current);} else {gama.logprior=0;}
  for(k=0;k<numtk;k++){if(f[k].propsd>verysmall) {f[k].logprior=priorparam(&f[k],f[k].current);} else {f[k].logprior=0;}}
  for(k=0;k<Number_Exp;++k){if(sigmap[k].propsd>verysmall) {sigmap[k].logprior=priorparam(&sigmap[k],sigmap[k].current);} else {sigmap[k].logprior=0;}}
  for(k=0;k<Number_Exp;++k){if(gamap[k].propsd>verysmall) {gamap[k].logprior=priorparam(&gamap[k],gamap[k].current);} else {gamap[k].logprior=0;}}
  
  logpost=lambda.logprior; logpost+=rho.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; logpost+=sigma.logprior; logpost+=gama.logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
  logprior=logpost;

  while(feasible==0) {
  feasible=1; // assume feasible, then look for 'problems' i.e. during likelihood calculations below, identify cases which shows that the current sequence has probability zero
  for(k=0;k<numobs;k++) // resample each infection time in turn (uniformly at random over the appropriate time interval
    {
      if(gobs[k]!=0)
		{
		a=infection_lower_bound[gobs[k]];
		b=infection_upper_bound[gobs[k]];
		unit[iobs[k]].infection_time=a+(b-a)*gsl_rng_uniform(r);
		}
    }

  for(i=0;i<nunits;i++) // loop over all units; compute likelihood contribution associated with each one
    {
	  if(unit[i].initial==0) // excludes initial colonies, which don't contribute to the likelihood (and note that initial susceptibilities/infectivities for ALL units were computed earlier)
	    {
	      if(unit[i].new==1)ttemp = unit[i].infection_time;
	      if(unit[i].new==0)ttemp = t_end;
	      unit[i].survival=0.0;
	      unit[i].hazard=0.0;

	      unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		  //printf("\nExternal survival: eps=%f intf from 0 to %f=%f",unit[i].eps,ttemp,intf(0,ttemp,tk));
		  if(unit[i].new==1)unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

		  // Loop over all potential donors to i: look in N(i)=I(i)
		  for(j=0;j<nbinfecsize[i];j++)
		  {
			  if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is really a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk);
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
		  }

	      unit[i].likelihood_contribution=0.0;//default, ie no contribution
	      if(unit[i].new==1)
		  {
			if(unit[i].hazard<verysmall) {feasible=0; printf("\nWARNING: Non-feasible time sequence identified...");}
			if(unit[i].susceptibility<verysmall) {feasible=0; printf("\nERROR: infection of unsusceptible unit..."); exit(0);}
			if(feasible==1) unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
		  }
	      if(unit[i].new==0)unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;
		  logpost+=unit[i].likelihood_contribution;
		  //printf("\nLlogL(%d)=%f",i,unit[i].likelihood_contribution);
	    }
		printf("\nLlogL(%d)=%f surv=%f haz=%f",i,unit[i].likelihood_contribution,unit[i].survival,unit[i].hazard);
		if(i==1930) printf("\nInf=%f Susc=%f LU=%f %f %f %f %f %f ...",unit[i].infectivity,unit[i].susceptibility,unit[i].h[0],unit[i].h[1],unit[i].h[2],unit[i].h[3],unit[i].h[4],unit[i].h[5]);
    }
  } // end of 'feasible' while loop

	printf("\nLogpost=%f   ",logpost);
	loglik=logpost-logprior;

  //----------------------------------------------------------------------------------------------------------
  //                                           Burnin
  //----------------------------------------------------------------------------------------------------------
  burninflag=1;
  mean_time_propsd=time_propsd;
  thin2=0;no_paramchanges=0;potential_paramchanges=0;no_timechanges=0;potential_timechanges=0;
  for(iteration=0;iteration<burnin;iteration++)
    {
      thin2++;
      if((PRINT2SCREEN)&&(thin2==thin2screen))
	  {
		  printf("\n  BI %d/%d, p(time change)=%.3f, p(param change)=%.3f, logL=%.0f, propsds= ",iteration+1,burnin,((double)(no_timechanges))/((double)(potential_timechanges)),((double)(no_paramchanges))/((double)(potential_paramchanges)),logpost);
		  printf("%f %f %f ",lambda.propsd, eps[1].propsd, rho.propsd);
		  if(sigma.propsd>verysmall) printf("%f ",sigma.propsd);
		  if(gama.propsd>verysmall) printf("%f ",gama.propsd);
		  for(i=0;i<Number_Eps;i++) {if(eps[i].propsd>verysmall) printf("%f ",eps[i].propsd);}
		  for(i=0;i<numtk;i++) {if(f[i].propsd>verysmall) printf("%f ",f[i].propsd);}
		  printf("%f ",mean_time_propsd);
		  thin2=0;
		  no_paramchanges=0;
		  potential_paramchanges=0;
		  no_timechanges=0;
		  potential_timechanges=0;
	  }

      //propose change to parameters
	  if(lambda.propsd>verysmall) samplelambda(r,tk);
	  if(rho.propsd>verysmall) samplerho(r,tk);
	  for(iteration2=0;iteration2<Number_Eps;iteration2++) { if(eps[iteration2].propsd>verysmall) sampleeps(iteration2,r,tk);}
	  if(sigma.propsd>verysmall) samplesigma(r,tk);
      if(gama.propsd>verysmall) samplegama(r,tk);
      for(iteration2=0;iteration2<numtk;iteration2++) { if(f[iteration2].propsd>verysmall) samplef(iteration2,r,tk);}
	  for(iteration2=0;iteration2<Number_Exp;iteration2++) { if(sigmap[iteration2].propsd>verysmall) samplesigmap(iteration2,r,tk);}
	  for(iteration2=0;iteration2<Number_Exp;iteration2++) { if(gamap[iteration2].propsd>verysmall) samplegamap(iteration2,r,tk);}

      //propose changes to infection times
      count=0;
      mean_time_propsd=0;
      for(i=0;i<nunits;++i)
	  {
	      if(unit[i].new)
		  {
			  sampletime(i,r,tk);
			  mean_time_propsd+=unit[i].tp/newcols;
			  if(iteration % SAVELATENT == 0) fprintf(lfptr,"%f ",unit[i].infection_time);
			  if(INCLUDETIMES) count++;
		  }
	  }
	  if(iteration % SAVELATENT == 0) fprintf(lfptr,"\n");
	  
	  fprintf(fptr,"%f %f ",lambda.current, rho.current);
	  for(kk=0;kk<Number_Eps;kk++) fprintf(fptr,"%f ",eps[kk].current);
	  for(kk=0;kk<Number_Simp;kk++) fprintf(fptr,"%f ",sigma.current[kk]);
	  for(kk=0;kk<Number_Simp;kk++) fprintf(fptr,"%f ",gama.current[kk]);
	  for(kk=0;kk<numtk;kk++) fprintf(fptr,"%f ",f[kk].current);
	  for(kk=0;kk<Number_Exp;kk++) fprintf(fptr,"%f ",sigmap[kk].current);
	  for(kk=0;kk<Number_Exp;kk++) fprintf(fptr,"%f ",gamap[kk].current);
	  fprintf(fptr,"%f ",logpost-logprior);
	  ilr();
	  ilr2();
	  for(kk=0;kk<Number_Simp-1;kk++) fprintf(fptr,"%f ",xi[kk].current);
	  for(kk=0;kk<Number_Simp-1;kk++) fprintf(fptr,"%f ",xi2[kk].current);
	  fprintf(fptr,"\n");

	  if(SAVEPROPSD)
	  {
		  fprintf(pfptr,"%f %f ",lambda.propsd,rho.propsd);
		  for(kk=0;kk<Number_Eps;kk++) fprintf(pfptr,"%f ",eps[kk].propsd);
		  fprintf(pfptr,"%f %f ",sigma.propsd,gama.propsd);
		  for(kk=0;kk<numtk;kk++) fprintf(pfptr,"%f ",f[kk].propsd);
		  for(kk=0;kk<Number_Exp;kk++) fprintf(pfptr,"%f ",sigmap[kk].propsd);
		  for(kk=0;kk<Number_Exp;kk++) fprintf(pfptr,"%f ",gamap[kk].propsd);
		  fprintf(pfptr,"\n");
	  }

    }

  if(SAVEPROPSD) fclose(pfptr);

  //----------------------------------------------------------------------------------------------------------
  //                                            MCMC
  //----------------------------------------------------------------------------------------------------------
  burninflag=0;
  thin2=0;no_paramchanges=0;potential_paramchanges=0;no_timechanges=0;potential_timechanges=0;
  for(iteration=0;iteration<mcmc;iteration++)
    {
	  thin2++;
      if((PRINT2SCREEN)&&(thin2==thin2screen))
	  {
		  printf("\n  MCMC %d/%d, p(time change)=%.3f, p(param change)=%.3f, logL=%.0f ",iteration+1,mcmc,((double)(no_timechanges))/((double)(potential_timechanges)),((double)(no_paramchanges))/((double)(potential_paramchanges)),logpost);
		  printf("lambda=%f f=(%f,%f) ",lambda.current,f[0].current,f[1].current);
		  thin2=0;
		  no_paramchanges=0;
		  potential_paramchanges=0;
		  no_timechanges=0;
		  potential_timechanges=0;
	  }

	  //propose change to parameters
	  if(lambda.propsd>verysmall) samplelambda(r,tk);
	  if(rho.propsd>verysmall) samplerho(r,tk);
	  for(iteration2=0;iteration2<Number_Eps;iteration2++) { if(eps[iteration2].propsd>verysmall) sampleeps(iteration2,r,tk);}
	  if(sigma.propsd>verysmall) samplesigma(r,tk);
      if(gama.propsd>verysmall) samplegama(r,tk);
      for(iteration2=0;iteration2<numtk;iteration2++) { if(f[iteration2].propsd>verysmall) samplef(iteration2,r,tk);}
	  for(iteration2=0;iteration2<Number_Exp;iteration2++) { if(sigmap[iteration2].propsd>verysmall) samplesigmap(iteration2,r,tk);}
	  for(iteration2=0;iteration2<Number_Exp;iteration2++) { if(gamap[iteration2].propsd>verysmall) samplegamap(iteration2,r,tk);}

	  //propose changes to infection times
      count=0;
      for(i=0;i<nunits;++i)
	  {
	      if(unit[i].new)
		  {
			  sampletime(i,r,tk);
			  if(iteration % SAVELATENT == 0) fprintf(lfptr,"%f ",unit[i].infection_time);
			  if(INCLUDETIMES) count++;
		  }
	  }
	  if(iteration % SAVELATENT == 0) fprintf(lfptr,"\n");

	  fprintf(fptr,"%f %f ",lambda.current, rho.current);
	  for(kk=0;kk<Number_Eps;kk++) fprintf(fptr,"%f ",eps[kk].current);
	  for(kk=0;kk<Number_Simp;kk++) fprintf(fptr,"%f ",sigma.current[kk]);
	  for(kk=0;kk<Number_Simp;kk++) fprintf(fptr,"%f ",gama.current[kk]);
	  for(kk=0;kk<numtk;kk++) fprintf(fptr,"%f ",f[kk].current);
	  for(kk=0;kk<Number_Exp;kk++) fprintf(fptr,"%f ",sigmap[kk].current);
	  for(kk=0;kk<Number_Exp;kk++) fprintf(fptr,"%f ",gamap[kk].current);
	  fprintf(fptr,"%f ",logpost-logprior);
	  ilr();
	  ilr2();
	  for(kk=0;kk<Number_Simp-1;kk++) fprintf(fptr,"%f ",xi[kk].current);
	  for(kk=0;kk<Number_Simp-1;kk++) fprintf(fptr,"%f ",xi2[kk].current);
	  fprintf(fptr,"\n");

}

fclose(fptr);
fclose(lfptr);
 return 0;

} // end of 'main' function

double kernel(int i, int j)//returns kernel rate from (0,0) to (i,j) parameterised by lambda.current
{
  double output=0.0, distance_squared;
  distance_squared = (double)(i*i) + (double)(j*j);
  if(distance_squared>0) output = pow(distance_squared,-1.0*lambda.current);//ie rate is d^{-2lambda}
  if(distance_squared>((double)dmax)*((double)dmax)) output = 0.0;
  return(output);
}

//these return the priors for the parameters
double priorsimp(double *v) // evaluate simplex prior, which is a Dirichlet distribution Dir(a_1,a_2,...,a_{Number_Simp}), with a_i=ALP (constant), at vector v
{
  double output=veryverysmalllogprior;
  int i,p;
  double s;
  double ap[Number_Simp];

  p=0;
  s=0;
  for(i=0;i<Number_Simp;i++) ap[i]=ALP;
  for(i=0;i<Number_Simp;i++) {s=s+v[i]; if((v[i]>verysmall) && (1-v[i]>verysmall)) p++;}
  if((p==Number_Simp)&&(abs(s-1)<0.0000000001)) output=gsl_ran_dirichlet_lnpdf(Number_Simp,ap,v);
  return(output);
}

double priorparam(parameter *p,double x) // evaluate prior for parameter p at value x
{
  double output=veryverysmalllogprior;

  if((x>=p->prior_l) && (x<=p->prior_u))
  {
	if(p->prior_e)
	{
		output=-p->prior_r*x; // assuming x has a truncated exponential distribution
	}
	else
	{
		output=0; // assuming x has a uniform distribution
	}
  }

  return(output);
}


//these are the functions to make proposals to the various parameters and decide whether or not to accept them
void samplesigma(gsl_rng *r,double *tk)
{
  double alph[Number_Simp];
  double alph2[Number_Simp];
  // Propose change to the sigma parameter vector used in the calculation of the susc/inf for units
  // Key point: this requires re-computing the susc/inf and hence also the unit-specific survival/hazard functions
  potential_paramchanges++;
  for(k=0;k<Number_Simp;k++) sigma.old[k]=sigma.current[k];//store current values
  for(k=0;k<Number_Simp; k++) alph[k]=delta+sigma.old[k]/sigma.propsd; // define concentration parameters
  gsl_ran_dirichlet(r, Number_Simp, alph, sigma.current);//propose change and store result in sigma.current
  for(k=0;k<Number_Simp;k++)
  {
	  if(sigma.current[k]<verysmall) printf("Warning: sigma.current[%d]=%.10f!!! but\n",k,sigma.current[k]);
	  if((sigma.current[k]>0)&&(sigma.current[k]<verysmall)) printf("BUT >0!!!");
  }
  for(k=0;k<Number_Simp; k++) alph2[k]=delta+sigma.current[k]/sigma.propsd;

 
  possible=1;

  if(possible)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      sigma.oldlogprior=sigma.logprior;
      a=priorsimp(sigma.current);
      if(a<verysmalllogprior)//if prior support for proposal too low then reject
	{
	  for(k=0;k<Number_Simp;k++) sigma.current[k]=sigma.old[k];
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  sigma.oldlogprior=sigma.logprior;
	  sigma.logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;
	  
	  for(i=0;i<nunits;i++)
	  {
		if(unit[i].initial==0)
		{
		  unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
		  unit[i].old_survival = unit[i].survival;
		  unit[i].old_hazard   = unit[i].hazard;
		  unit[i].hazard=0;
		  unit[i].survival=0;
		}
		  unit[i].old_susceptibility=unit[i].susceptibility;
		  unit[i].old_infectivity=unit[i].infectivity;
		  unit[i].susceptibility = 0.0;
		  for(k=0;k<Number_Simp;++k) unit[i].susceptibility += sigma.current[k]*unit[i].h[k];
		  exponent=0;
		  for(k=0;k<Number_Exp;k++) exponent+=sigmap[k].current*unit[i].k[k];
		  unit[i].susceptibility*=exp(exponent);
	  }

	  for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		    if(unit[i].new==1) unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

			for(j=0;j<nbinfecsize[i];j++)
			{
				if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk); // NB varying infectivity due to incorporating suitability of unit j
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
			}

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }

  	  prob=log(gsl_rng_uniform(r));
	  pacc=logpost-old_logpost+gsl_ran_dirichlet_lnpdf(Number_Simp,alph2,sigma.old)-gsl_ran_dirichlet_lnpdf(Number_Simp,alph,sigma.current);
	  no_paramchanges++;
	  if(burninflag) sigma.propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) sigma.propsd*=f2/f1;
	      no_paramchanges--;
		  for(k=0;k<Number_Simp;k++) sigma.current[k]=sigma.old[k];
	      logpost=old_logpost;
		  logprior=old_logprior;
	      sigma.logprior=sigma.oldlogprior;
	      for(i=0;i<nunits;++i)
		  {
			unit[i].susceptibility=unit[i].old_susceptibility;
			unit[i].infectivity=unit[i].old_infectivity;
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
		    }
		  }
	    }
	}

    }
  if(!possible) {for(k=0;k<Number_Simp;k++) sigma.current[k]=sigma.old[k]; if(burninflag) sigma.propsd*=f2;}

}

void samplegama(gsl_rng *r,double *tk)
{
  double alph[Number_Simp];
  double alph2[Number_Simp];
  // Propose change to the gama parameter vector used in the calculation of the susc/inf for units
  // Key point: this requires re-computing the susc/inf and hence also the unit-specific survival/hazard functions
  potential_paramchanges++;
  for(k=0;k<Number_Simp;k++) gama.old[k]=gama.current[k];//store current values
  for(k=0;k<Number_Simp; k++) alph[k]=delta+gama.old[k]/gama.propsd; // define concentration parameters
  gsl_ran_dirichlet(r, Number_Simp, alph, gama.current);//propose change and store result in gama.current
  for(k=0;k<Number_Simp; k++) alph2[k]=delta+gama.current[k]/gama.propsd;

  possible=1;

  if(possible)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      gama.oldlogprior=gama.logprior;
      a=priorsimp(gama.current);
      if(a<verysmalllogprior)//if prior support for proposal too low then reject
	{
	  for(k=0;k<Number_Simp;k++) gama.current[k]=gama.old[k];
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  gama.oldlogprior=gama.logprior;
	  gama.logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;
	  
	  for(i=0;i<nunits;i++)
	  {
		if(unit[i].initial==0)
		{
		  unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
		  unit[i].old_survival = unit[i].survival;
		  unit[i].old_hazard   = unit[i].hazard;
		  unit[i].hazard=0;
		  unit[i].survival=0;
		}
		  unit[i].old_susceptibility=unit[i].susceptibility;
		  unit[i].old_infectivity=unit[i].infectivity;
		  unit[i].infectivity = 0.0;
		  for(k=0;k<Number_Simp;++k) unit[i].infectivity += gama.current[k]*unit[i].h[k];
		  exponent=0;
		  for(k=0;k<Number_Exp;k++) exponent+=gamap[k].current*unit[i].k[k];
		  unit[i].infectivity*=exp(exponent);
	  }

	  for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		    if(unit[i].new==1) unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

			for(j=0;j<nbinfecsize[i];j++)
			{
				if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk); // NB varying infectivity due to incorporating suitability of unit j
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
			}

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }

  	  prob=log(gsl_rng_uniform(r));
	  pacc=logpost-old_logpost+gsl_ran_dirichlet_lnpdf(Number_Simp,alph2,gama.old)-gsl_ran_dirichlet_lnpdf(Number_Simp,alph,gama.current);
	  no_paramchanges++;
	  if(burninflag) gama.propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) gama.propsd*=f2/f1;
	      no_paramchanges--;
		  for(k=0;k<Number_Simp;k++) gama.current[k]=gama.old[k];
	      logpost=old_logpost;
		  logprior=old_logprior;
	      gama.logprior=gama.oldlogprior;
	      for(i=0;i<nunits;++i)
		  {
			unit[i].susceptibility=unit[i].old_susceptibility;
			unit[i].infectivity=unit[i].old_infectivity;
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
		    }
		  }
	    }
	}

    }
  if(!possible) {for(k=0;k<Number_Simp;k++) gama.current[k]=gama.old[k]; if(burninflag) gama.propsd*=f2;}

}

void samplesigmap(int index,gsl_rng *r,double *tk)
{
  // Proposes change to the exponential parameters in the susceptibility function.
  potential_paramchanges++;
  sigmap[index].old=sigmap[index].current;
  sigmap[index].current+=sigmap[index].propsd*rn(r);

  possible=1;// NB can put some other conditions here, eg to force the parameter to be positive
  if(possible)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      a=priorparam(&sigmap[index],sigmap[index].current);
      if(a<verysmalllogprior)//if prior support for proposal too low then reject
	{
	  sigmap[index].current=sigmap[index].old;
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  sigmap[index].oldlogprior=sigmap[index].logprior;
	  sigmap[index].logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;
	  
	  for(i=0;i<nunits;i++)
	  {
		if(unit[i].initial==0)
		{
		  unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
		  unit[i].old_survival = unit[i].survival;
		  unit[i].old_hazard   = unit[i].hazard;
		  unit[i].hazard=0;
		  unit[i].survival=0;
		}
		unit[i].old_susceptibility=unit[i].susceptibility;
		unit[i].old_infectivity=unit[i].infectivity;
		unit[i].susceptibility = 0.0;
		for(k=0;k<Number_Simp;++k) unit[i].susceptibility += sigma.current[k]*unit[i].h[k];
		exponent=0;
		for(k=0;k<Number_Exp;k++) exponent+=sigmap[k].current*unit[i].k[k];
		unit[i].susceptibility*=exp(exponent);
	  }

	  for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		    if(unit[i].new==1) unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

			for(j=0;j<nbinfecsize[i];j++)
			{
				if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is really a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk); // NB varying infectivity due to incorporating suitability of unit j
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
			}

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }


	  prob=log(gsl_rng_uniform(r));
	  pacc=logpost-old_logpost;
	  no_paramchanges++;
	  if(burninflag) sigmap[index].propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) sigmap[index].propsd*=f2/f1;
	      no_paramchanges--;
	      sigmap[index].current=sigmap[index].old;
	      logpost=old_logpost;
		  logprior=old_logprior;
	      sigmap[index].logprior=sigmap[index].oldlogprior;
		  for(i=0;i<nunits;i++)
		  {
			unit[i].susceptibility=unit[i].old_susceptibility;
			unit[i].infectivity=unit[i].old_infectivity;
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
		    }
		  }
	    }
	}
    }
  if(possible==0){sigmap[index].current=sigmap[index].old; if(burninflag) sigmap[index].propsd*=f2;}

}

void samplegamap(int index,gsl_rng *r,double *tk)
{
  // Proposes change to the exponential parameters in the susceptibility function.
  potential_paramchanges++;
  gamap[index].old=gamap[index].current;
  gamap[index].current+=gamap[index].propsd*rn(r);

  possible=1;// NB can put some other conditions here, eg to force the parameter to be positive
  if(possible)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      a=priorparam(&gamap[index],gamap[index].current);
      if(a<verysmalllogprior)//if prior support for proposal too low then reject
	{
	  gamap[index].current=gamap[index].old;
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  gamap[index].oldlogprior=gamap[index].logprior;
	  gamap[index].logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;
	  
	  for(i=0;i<nunits;i++)
	  {
		if(unit[i].initial==0)
		{
		  unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
		  unit[i].old_survival = unit[i].survival;
		  unit[i].old_hazard   = unit[i].hazard;
		  unit[i].hazard=0;
		  unit[i].survival=0;
		}
		unit[i].old_susceptibility=unit[i].susceptibility;
		unit[i].old_infectivity=unit[i].infectivity;
		unit[i].infectivity = 0.0;
		for(k=0;k<Number_Simp;++k) unit[i].infectivity += gama.current[k]*unit[i].h[k];
		exponent=0;
		for(k=0;k<Number_Exp;k++) exponent+=gamap[k].current*unit[i].k[k];
		unit[i].infectivity*=exp(exponent);
	  }

	  for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		    if(unit[i].new==1) unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

			for(j=0;j<nbinfecsize[i];j++)
			{
				if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is really a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk); // NB varying infectivity due to incorporating suitability of unit j
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
			}

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }


	  prob=log(gsl_rng_uniform(r));
	  pacc=logpost-old_logpost;
	  no_paramchanges++;
	  if(burninflag) gamap[index].propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) gamap[index].propsd*=f2/f1;
	      no_paramchanges--;
	      gamap[index].current=gamap[index].old;
	      logpost=old_logpost;
		  logprior=old_logprior;
	      gamap[index].logprior=gamap[index].oldlogprior;
		  for(i=0;i<nunits;i++)
		  {
			unit[i].susceptibility=unit[i].old_susceptibility;
			unit[i].infectivity=unit[i].old_infectivity;
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
		    }
		  }
	    }
	}
    }
  if(possible==0){gamap[index].current=gamap[index].old; if(burninflag) gamap[index].propsd*=f2;}

}


void samplerho(gsl_rng *r,double *tk)
{
	// Proposes change to the rho parameter
  potential_paramchanges++;
  rho.old=rho.current;
  rho.current+=rho.propsd*rn(r);

  possible=1;// NB can put some other conditions here, eg to force the parameter to be positive
  if(possible)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      rho.oldlogprior=rho.logprior;
      a=priorparam(&rho,rho.current);
      if(a<verysmalllogprior)//if prior support for proposal too low then reject
	{
	  rho.current=rho.old;
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  rho.oldlogprior=rho.logprior;
	  rho.logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;

	  for(i=0;i<nunits;i++)
	  {
		if(unit[i].initial==0)
		{
		  unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
		  unit[i].old_survival = unit[i].survival;
		  unit[i].old_hazard   = unit[i].hazard;
		  unit[i].hazard=0;
		  unit[i].survival=0;
		}
	  }

	  for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		    if(unit[i].new==1) unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

			for(j=0;j<nbinfecsize[i];j++)
			{
				if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is really a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk); // NB varying infectivity due to incorporating suitability of unit j
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
			}

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }


	  prob=log(gsl_rng_uniform(r));
	  pacc=logpost-old_logpost;
	  no_paramchanges++;
	  if(burninflag) rho.propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) rho.propsd*=f2/f1;
	      no_paramchanges--;
	      rho.current=rho.old;
	      logpost=old_logpost;
		  logprior=old_logprior;
	      rho.logprior=rho.oldlogprior;
		  for(i=0;i<nunits;i++)
		  {
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
		    }
		  }
	    }
	}
    }
  if(possible==0){rho.current=rho.old; if(burninflag) rho.propsd*=f2;}
}

void samplelambda(gsl_rng *r,double *tk)
{
  // Proposes change to the lambda parameter in the kernel; this requires a full re-compute of the likelihood (except for the suitability functions)
  potential_paramchanges++;
  lambda.old=lambda.current;
  lambda.current+=lambda.propsd*rn(r);

  possible=0;
  if(lambda.current>verysmall) possible=1;
  if(possible)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      lambda.oldlogprior=lambda.logprior;

      a=priorparam(&lambda,lambda.current);
      if(a<verysmalllogprior)
	{
	  lambda.current=lambda.old;
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  lambda.oldlogprior=lambda.logprior;
	  lambda.logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;

	  // Compute kernel to get kernel matrix
	  b=0.0;
	  for(x_distance=0;x_distance<=dmax;++x_distance)
	  {
	    for(y_distance=0;y_distance<=dmax;++y_distance)
		{
		  old_kernel_matrix[x_distance][y_distance]=kernel_matrix[x_distance][y_distance];
		  kernel_matrix[x_distance][y_distance]=kernel(x_distance,y_distance);
		  if((x_distance==0)&&(y_distance!=0))b=b+2.0*kernel_matrix[x_distance][y_distance];
		  if((x_distance!=0)&&(y_distance==0))b=b+2.0*kernel_matrix[x_distance][y_distance];
		  if((x_distance!=0)&&(y_distance!=0))b=b+4.0*kernel_matrix[x_distance][y_distance];
		}
	  }
	  for(x_distance=0;x_distance<=dmax;++x_distance){for(y_distance=0;y_distance<=dmax;++y_distance)kernel_matrix[x_distance][y_distance]=kernel_matrix[x_distance][y_distance]/b;}// renormalises the kernel_matrix

	  // Compute link strengths
	  for(i=0;i<nunits;i++)
	  {
		for(j=0;j<nbsize[i];j++) linknb[i][j]=kernel_matrix[abs(unit[i].x-unit[nb[i][j]].x)][abs(unit[i].y-unit[nb[i][j]].y)];
	    for(j=0;j<nbinfecsize[i];j++) {linknbinfec[i][j]=kernel_matrix[abs(unit[i].x-unit[nbinfec[i][j]].x)][abs(unit[i].y-unit[nbinfec[i][j]].y)]; revlinknbinfec[i][j]=linknbinfec[i][j];}
	    for(j=0;j<infecsize[i];j++) linkinfec[i][j]=kernel_matrix[abs(unit[i].x-unit[infec[i][j]].x)][abs(unit[i].y-unit[infec[i][j]].y)];
	  }

	   for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			unit[i].eps_old=unit[i].eps;
			unit[i].eps=epscalc(i);
			unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
		    unit[i].old_survival = unit[i].survival;
		    unit[i].old_hazard   = unit[i].hazard;
			unit[i].hazard=0;
		    unit[i].survival=0;
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		    if(unit[i].new==1) unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

			for(j=0;j<nbinfecsize[i];j++)
			{
				if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is really a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk); // NB varying infectivity due to incorporating suitability of unit j
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
			}

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }

	  prob=log(gsl_rng_uniform(r));
	  pacc=logpost-old_logpost;
	  no_paramchanges++;
	  if(burninflag) lambda.propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) lambda.propsd*=f2/f1;
	      no_paramchanges--;
	      lambda.current=lambda.old;
	      logpost=old_logpost;
		  logprior=old_logprior;
	      lambda.logprior=lambda.oldlogprior;
		  for(x_distance=0;x_distance<=dmax;++x_distance){for(y_distance=0;y_distance<=dmax;++y_distance)kernel_matrix[x_distance][y_distance]=old_kernel_matrix[x_distance][y_distance];}
	      for(i=0;i<nunits;++i)
		  {
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
			  unit[i].eps=unit[i].eps_old;
		    }
			for(j=0;j<nbsize[i];j++) linknb[i][j]=kernel_matrix[abs(unit[i].x-unit[nb[i][j]].x)][abs(unit[i].y-unit[nb[i][j]].y)];
	        for(j=0;j<nbinfecsize[i];j++) {linknbinfec[i][j]=kernel_matrix[abs(unit[i].x-unit[nbinfec[i][j]].x)][abs(unit[i].y-unit[nbinfec[i][j]].y)]; revlinknbinfec[i][j]=linknbinfec[i][j];}
	        for(j=0;j<infecsize[i];j++) linkinfec[i][j]=kernel_matrix[abs(unit[i].x-unit[infec[i][j]].x)][abs(unit[i].y-unit[infec[i][j]].y)];
		  }
	    }
	}
    }

  if(possible==0) {lambda.current=lambda.old; if(burninflag) lambda.propsd*=f2;}


}

void sampleeps(int index, gsl_rng *r,double *tk)
{
  // Proposes change to an epsilon parameter (background transmission rate); requires adjustments to survival/hazard functions
  potential_paramchanges++;
  eps[index].old=eps[index].current;
  eps[index].current+=eps[index].propsd*rn(r);
  possible=0;
  if(eps[index].current>verysmall)possible=1;  // check that proposed parameter value is still >0
  if(possible==1)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      a=priorparam(&eps[index],eps[index].current);
      if(a<verysmalllogprior)
	{
	  eps[index].current=eps[index].old;
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  eps[index].oldlogprior=eps[index].logprior;
	  eps[index].logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;

	  for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			unit[i].eps_old=unit[i].eps;
			unit[i].eps=epscalc(i); // NB depends on the unit i
			unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
			unit[i].old_survival = unit[i].survival;
			unit[i].old_hazard   = unit[i].hazard;
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  (unit[i].eps-unit[i].eps_old)*intpf(0,ttemp,tk);
		    unit[i].hazard += (unit[i].eps-unit[i].eps_old)*evalpf(ttemp,tk); // this should only be done if unit[i].new==1... but the following line corrects the situation

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }

	  prob=gsl_rng_uniform(r);
	  pacc=exp(logpost-old_logpost);
	  no_paramchanges++;
	  if(burninflag) eps[index].propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) eps[index].propsd*=f2/f1;
	      no_paramchanges--;
		  eps[index].current=eps[index].old;
	      logpost=old_logpost;
		  logprior=old_logprior;
	      eps[index].logprior=eps[index].oldlogprior;
		  for(i=0;i<nunits;++i)
		  {
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
			  unit[i].eps=unit[i].eps_old;
		    }
		  }
	    }

	}
    }
  if(possible==0) {eps[index].current=eps[index].old; if(burninflag) eps[index].propsd*=f2;}


}

void sampletime(int i, gsl_rng *r,double *tk)
{
  // Proposes change to one of the infection times
  potential_timechanges++;
  int obst;
  double oldt,newt;
  int feasible; // catch proposals which lead to moves to infeasible time sequences before numerical issues are encountered
  obst=unit[i].timeclass;
  unit[i].old_infection_time=unit[i].infection_time;

  unit[i].infection_time+=unit[i].tp*rn(r);

  possible=0;
  if((unit[i].infection_time>infection_lower_bound[obst])&&(unit[i].infection_time<infection_upper_bound[obst]))possible=1;
  if(possible)
    {
      feasible=1; // assume feasible then look for problems during calculations
      oldt=unit[i].old_infection_time;
      newt=unit[i].infection_time;

	  for(j=0;j<nunits;++j) // store old data: could restrict this to N(i) U I(i) but not yet convinced this is worth the effort
	  {
		if(unit[j].initial==0)
		{
		  unit[j].old_likelihood_contribution=unit[j].likelihood_contribution;
		  unit[j].old_survival = unit[j].survival;
		  unit[j].old_hazard   = unit[j].hazard;
		}
	  }

      old_logpost=logpost;
	  old_logprior=logprior;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;

	   for(j=0;j<nbinfecsize[i];j++) // loop through j\in N(i)^I(i)
	  {
		  if(unit[nbinfec[i][j]].new)
		  {
			  if((newt<unit[nbinfec[i][j]].infection_time) && (oldt<unit[nbinfec[i][j]].infection_time)) // (1)
			  {
				  unit[nbinfec[i][j]].survival-=rho.current*revlinknbinfec[i][j]*intf(oldt,newt,tk)*unit[i].infectivity;
			  }
			  if((newt>unit[nbinfec[i][j]].infection_time) && (oldt>unit[nbinfec[i][j]].infection_time)) // (2)
			  {
				  unit[i].survival+=rho.current*linknbinfec[i][j]*intf(oldt,newt,tk)*unit[nbinfec[i][j]].infectivity;
			  }
			  if((newt>unit[nbinfec[i][j]].infection_time) && (oldt<unit[nbinfec[i][j]].infection_time)) // (3)
			  {
				  unit[i].survival+=rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,newt,tk)*unit[nbinfec[i][j]].infectivity;
				  unit[i].hazard+=rho.current*linknbinfec[i][j]*unit[nbinfec[i][j]].infectivity*evalf(newt,tk);
				  unit[nbinfec[i][j]].survival-=rho.current*revlinknbinfec[i][j]*intf(oldt,unit[nbinfec[i][j]].infection_time,tk)*unit[i].infectivity;
				  unit[nbinfec[i][j]].hazard-=rho.current*revlinknbinfec[i][j]*unit[i].infectivity*evalf(unit[nbinfec[i][j]].infection_time,tk);
			  }
			  if((newt<unit[nbinfec[i][j]].infection_time) && (oldt>unit[nbinfec[i][j]].infection_time)) // (4)
			  {
				  unit[i].survival-=rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,oldt,tk)*unit[nbinfec[i][j]].infectivity;
				  unit[i].hazard-=rho.current*linknbinfec[i][j]*unit[nbinfec[i][j]].infectivity*evalf(newt,tk);
				  unit[nbinfec[i][j]].survival+=rho.current*revlinknbinfec[i][j]*intf(newt,unit[nbinfec[i][j]].infection_time,tk)*unit[i].infectivity;
				  unit[nbinfec[i][j]].hazard+=rho.current*revlinknbinfec[i][j]*unit[i].infectivity*evalf(unit[nbinfec[i][j]].infection_time,tk);
			  }
			  if(unit[nbinfec[i][j]].hazard<verysmall) {feasible=0; printf(" W");}
		      if(feasible) unit[nbinfec[i][j]].likelihood_contribution=log(unit[nbinfec[i][j]].hazard*unit[nbinfec[i][j]].susceptibility)-unit[nbinfec[i][j]].survival*unit[nbinfec[i][j]].susceptibility;
		  }
		  if((unit[nbinfec[i][j]].new==0) && (unit[nbinfec[i][j]].initial==0))
		  {
			  unit[nbinfec[i][j]].survival-=intf(oldt,newt,tk)*rho.current*revlinknbinfec[i][j]*unit[i].infectivity;
			  unit[nbinfec[i][j]].likelihood_contribution=-1.0*unit[nbinfec[i][j]].survival*unit[nbinfec[i][j]].susceptibility;
		  }
		  if(unit[nbinfec[i][j]].initial)
		  {
			  unit[i].survival+=intf(oldt,newt,tk)*rho.current*linknbinfec[i][j]*unit[nbinfec[i][j]].infectivity;
		  }
	  }

      unit[i].survival += unit[i].eps*intpf(oldt,newt,tk);


      if(unit[i].new)
	{
		if(unit[i].hazard<verysmall) {feasible=0; printf(" W");}
		if(feasible) unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
	}

	   for(j=0;j<nunits;j++)
	  {
		  if(unit[j].initial==0)
		  {
			  logpost+=unit[j].likelihood_contribution;
		  }
	  }
      prob=log(gsl_rng_uniform(r));
      pacc=logpost-old_logpost;
      no_timechanges++;
      if(burninflag) unit[i].tp*=f1;
      if((prob>pacc) || (feasible==0))//ie reject change
	{
	  if(burninflag) unit[i].tp*=f2/f1;
	  no_timechanges--;
	  unit[i].infection_time = unit[i].old_infection_time;
	  logpost = old_logpost;
	  logprior=old_logprior;
	  for(j=0;j<nunits;++j)
	  {
		if(unit[j].initial==0)
		{
		  unit[j].likelihood_contribution=unit[j].old_likelihood_contribution;
		  unit[j].survival = unit[j].old_survival;
		  unit[j].hazard   = unit[j].old_hazard;
		}
	  }
	}

    }
  if(possible==0){unit[i].infection_time = unit[i].old_infection_time; if(burninflag) unit[i].tp*=f2;}
}

void samplef(int index, gsl_rng *r,double *tk) // sample the parameter f[index] index=0...<numtk
{
  // Propose change to one of the f parameters used to define the time varying transmission rate function
  // Note: this requires re-computing the unit-specific survival/hazard functions
  potential_paramchanges++;
  for(k=0;k<numtk;k++) f[k].old=f[k].current;//store current values
  f[index].current+=f[index].propsd*rn(r);//propose change

  possible=0;
  if(f[index].current>verysmall)possible=1;  // checks whether proposal is >0

  if(possible)
    {
      old_logpost=logpost;
	  old_logprior=logprior;
      f[index].oldlogprior=f[index].logprior;
      a=priorparam(&f[index],f[index].current);
      if(a<verysmalllogprior)//if prior support for proposal too low then reject
	{
	  f[index].current=f[index].old;
	  logpost=old_logpost;
	  logprior=old_logprior;
	}
      if(a>verysmalllogprior)
	{
	  f[index].oldlogprior=f[index].logprior;
	  f[index].logprior=a;
	  logpost=lambda.logprior+rho.logprior+sigma.logprior+gama.logprior; for(k=0;k<Number_Eps;k++) logpost+=eps[k].logprior; for(k=0;k<numtk;k++) logpost+=f[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=sigmap[k].logprior; for(k=0;k<Number_Exp;k++) logpost+=gamap[k].logprior;
	  logprior=logpost;
	  for(i=0;i<nunits;i++)
	  {
		if(unit[i].initial==0)
		{
		  unit[i].old_likelihood_contribution=unit[i].likelihood_contribution;
		  unit[i].old_survival = unit[i].survival;
		  unit[i].old_hazard   = unit[i].hazard;
		  unit[i].hazard=0;
		  unit[i].survival=0;
		}
	  }

	  for(i=0;i<nunits;i++)
	  {
		  if(unit[i].initial==0)
		  {
			if(unit[i].new==1)ttemp = unit[i].infection_time;
			if(unit[i].new==0)ttemp = t_end;

			unit[i].survival +=  unit[i].eps*intpf(0,ttemp,tk);
		    if(unit[i].new==1) unit[i].hazard += unit[i].eps*evalpf(ttemp,tk);

			for(j=0;j<nbinfecsize[i];j++)
			{
				if((unit[nbinfec[i][j]].initial==1)||((unit[nbinfec[i][j]].new==1)&&(unit[nbinfec[i][j]].infection_time<ttemp)))//ie nb[i][j] is a potential donor of i
			    {
			      unit[i].survival += unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*intf(unit[nbinfec[i][j]].infection_time,ttemp,tk); // NB varying infectivity due to incorporating suitability of unit j
			      if(unit[i].new==1)unit[i].hazard+=unit[nbinfec[i][j]].infectivity*rho.current*linknbinfec[i][j]*evalf(ttemp,tk);
			    }
			}

			if(unit[i].new==0){unit[i].hazard=0.0;unit[i].likelihood_contribution=-1.0*unit[i].survival*unit[i].susceptibility;}
			if(unit[i].new==1)unit[i].likelihood_contribution=log(unit[i].hazard*unit[i].susceptibility)-unit[i].survival*unit[i].susceptibility;
			logpost+=unit[i].likelihood_contribution;
		  }
	  }

  	  prob=log(gsl_rng_uniform(r));
	  pacc=logpost-old_logpost;
	  no_paramchanges++;
	  if(burninflag) f[index].propsd*=f1;
	  if(prob>pacc)//ie reject change
	    {
	      if(burninflag) f[index].propsd*=f2/f1;
	      no_paramchanges--;
		  f[index].current=f[index].old;
	      logpost=old_logpost;
		  logprior=old_logprior;
	      f[index].logprior=f[index].oldlogprior;
	      for(i=0;i<nunits;++i)
		  {
		    if(unit[i].initial==0)
		    {
		      unit[i].likelihood_contribution=unit[i].old_likelihood_contribution;
			  unit[i].survival = unit[i].old_survival;
			  unit[i].hazard   = unit[i].old_hazard;
		    }
		  }
	    }
	}

    }
  if(!possible) {f[index].current=f[index].old; if(burninflag) f[index].propsd*=f2;}

}

double rn(gsl_rng *r)//generates a N(0,1) variate
{
  double out, I, J;
  I = gsl_rng_uniform(r);
  J = gsl_rng_uniform(r);
  out = sqrt(-2.0*log(I))*sin(2.0*PI*J);
  return(out);
}

double intpf(double u1, double u2, double *tk) // integrate time varying function phi(f(t)) from t=u1 to t=u2: currently phi is sqrt
{
	double out=0;
	int k;
	double u2p,u1p;

	if((u1<0) || (u2>t_end)) {printf("\nError with integration limits! \n"); exit(1);} // check integration limits are reasonable

	if(u1<u2)
	{
		out+=fmin(tk[0],u2)-fmin(tk[0],fmin(u2,u1)); // first time segment from 0 to tk[0] NB u1=max(u1,0) and phi(f)=1 on this segment

		for(k=0;k<numtk;k++) // loop over all numtk subintervals on the time axis following the initial segment
		{
			//out+=sqrt(f[k].current)*(fmin(tk[k+1],u2)-fmin(tk[k+1],fmin(u2,fmax(tk[k],u1)))); // phi=sqrt
			out+=f[k].current*(fmin(tk[k+1],u2)-fmin(tk[k+1],fmin(u2,fmax(tk[k],u1))));
		}
	}

	if(u1>u2)
	{
		u1p=u2;
		u2p=u1;
		out-=fmin(tk[0],u2p)-fmin(tk[0],fmin(u2p,u1p)); // first time segment from 0 to tk[0] NB u1=max(u1,0) and f=1 on this segment

		for(k=0;k<numtk;k++) // loop over all numtk subintervals on the time axis following the initial segment
		{
			//out-=sqrt(f[k].current)*(fmin(tk[k+1],u2p)-fmin(tk[k+1],fmin(u2p,fmax(tk[k],u1p)))); // phi=sqrt
			out-=f[k].current*(fmin(tk[k+1],u2p)-fmin(tk[k+1],fmin(u2p,fmax(tk[k],u1p))));
		}
	}

	return(out);
}

double intf(double u1, double u2, double *tk) // integrate time varying function f(t) from t=u1 to t=u2
{
	double out=0;
	int k;
	double u2p,u1p;

	if((u1<0) || (u2>t_end)) {printf("\nError with integration limits! \n"); exit(1);} // check integration limits are reasonable

	if(u1<u2)
	{
		out+=fmin(tk[0],u2)-fmin(tk[0],fmin(u2,u1)); // first time segment from 0 to tk[0] NB u1=max(u1,0) and f=1 on this segment

		for(k=0;k<numtk;k++) // loop over all numtk subintervals on the time axis following the initial segment
		{
			out+=f[k].current*(fmin(tk[k+1],u2)-fmin(tk[k+1],fmin(u2,fmax(tk[k],u1))));
		}
	}

	if(u1>u2)
	{
		u1p=u2;
		u2p=u1;
		out-=fmin(tk[0],u2p)-fmin(tk[0],fmin(u2p,u1p)); // first time segment from 0 to tk[0] NB u1=max(u1,0) and f=1 on this segment

		for(k=0;k<numtk;k++) // loop over all numtk subintervals on the time axis following the initial segment
		{
			out-=f[k].current*(fmin(tk[k+1],u2p)-fmin(tk[k+1],fmin(u2p,fmax(tk[k],u1p))));
		}
	}

	return(out);
}

double evalf(double u,double *tk) // evaluate time varying function f(t) at a specific t=u
{
	double out=1;
	int k;

	for(k=0;k<numtk;k++) {if(u>tk[k]) out=f[k].current;}

	return(out);
}

double evalpf(double u,double *tk) // evaluate ime varying function phi(f(t)) at a specific t=u
{
	double out=1;
	int k;

	//for(k=0;k<numtk;k++) {if(u>tk[k]) out=sqrt(f[k].current);}
	for(k=0;k<numtk;k++) {if(u>tk[k]) out=f[k].current;}

	return(out);
}

void ilr() // update xi (Number_Simp-1 dimensions) in real space following a change to the associated vector sigma in the Number_Simp dimensional simplex
{
	double logb[MNCOV]={0}; // temp storage for log(sigma)
	int k;
	double lg=0; // log(product of sigma's)

	for(k=0;k<Number_Simp;k++) logb[k]=log(sigma.current[k]);
	for(k=0;k<Number_Simp-1;k++)
	{
		lg+=logb[k];
		xi[k].current=sqrt((k+1.0000)/(k+2.0000))*(lg/(k+1.0000)-logb[k+1]);
	}
}

void ilr2() // update xi2 (Number_Simp-1 dimensions) in real space following a change to the associated vector gama in the Number_Simp dimensional simplex
{
	double logb[MNCOV]={0}; // temp storage for log(beta)
	int k;
	double lg=0; // log(product of beta's)

	for(k=0;k<Number_Simp;k++) logb[k]=log(gama.current[k]);
	for(k=0;k<Number_Simp-1;k++)
	{
		lg+=logb[k];
		xi2[k].current=sqrt((k+1.0000)/(k+2.0000))*(lg/(k+1.0000)-logb[k+1]);
	}
}

void display_help(void)
{
  printf("\nUsage: inference_vi  -r  <rep number 1..5>\n");
}

void get_arguments(int argc,char *argv[],int *repno)
{
	int ii;
	for(ii=0;ii<argc;ii++)
	    {
	     if(strcmp(argv[ii],"-r")==0) *repno=atoi(argv[ii+1]);
		}
}

void read_parameters(void)
{
	FILE *ptr;
	double dum[5];
	char pname[100];
	
	ptr=fopen("parameters.txt","r");
	
	readint(&Number_Simp,"Number_Simp",ptr);
	readint(&Number_Eps,"Number_Eps",ptr);
	readint(&Number_Exp,"Number_Exp",ptr);
	readint(&nunits,"nunits",ptr);
	readint(&nsubints,"nsi",ptr);
		
	fgets(line,500,ptr);
	if(strcmp(line,"subinttimes\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<nsubints;i++) fscanf(ptr,"%lf",&subinttimes[i]);
	fscanf(ptr,"\n");
	
	readint(&model,"model",ptr);
	readint(&numtk,"numtk",ptr);
	if(numtk>1000) {printf("\nError! tk too big."); exit(0);}
	
	fgets(line,500,ptr);
	if(strcmp(line,"tk\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<=numtk;i++) fscanf(ptr,"%lf",&tk[i]);
	fscanf(ptr,"\n");
	
	readint(&dmax,"maxdisp",ptr);
	readint(&burnin,"burnin",ptr);
	readint(&mcmc,"mcmc",ptr);
	
	readparam(&lambda,"lambda",ptr);
	readparam(&rho,"rho",ptr);
	vreadparam(&sigma,"sigma",ptr);
	vreadparam(&gama,"gamma",ptr);
	
	for(j=1;j<=Number_Eps;j++)
	{
		sprintf(pname,"epsilon[[%d]]",j);
		readparam(&eps[j-1],pname,ptr);
	}
	
	fgets(line,500,ptr);
	if(strcmp(line,"timev\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<=5;i++) fscanf(ptr,"%lf",&timev[i]);
	
	fclose(ptr);
}

void readint(int *rr,char *s,FILE *ptr)
{
	char ss[10000];
	
	sprintf(ss,"%s\n",s);
	fgets(line,500,ptr);
	if(strcmp(line,ss)!=0) {printf("\nFile input error for line=%s... and ss=%s... and diff=%d and lengths %d and %d\n",line,ss,strcmp(line,ss),strlen(line),strlen(ss)); exit(0);}
	fscanf(ptr,"%d\n",rr);
}


void readparam(parameter *p,char *s,FILE *ptr)
{
	char ss[10000];
	double dum[5];
	
	sprintf(ss,"%s\n",s);
	
	fgets(line,500,ptr);
	if(strcmp(line,ss)!=0) {printf("\nFile input error for line=%s.... and ss=%s...",line,ss); exit(0);}
	fgets(line,500,ptr);
	if(strcmp(line,"initial\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<5;i++) fscanf(ptr,"%lf",&dum[i]);
	fscanf(ptr,"\n");
	p->current=dum[repno-1]; // repno is 1..5
	fgets(line,500,ptr);
	if(strcmp(line,"propsd\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<5;i++) fscanf(ptr,"%lf",&dum[i]);
	fscanf(ptr,"\n");
	p->propsd=dum[repno-1]; // repno is 1..5
	fgets(line,500,ptr);
	if(strcmp(line,"prior_e\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	fscanf(ptr,"%d\n",&p->prior_e);
	fgets(line,500,ptr);
	if(strcmp(line,"prior_r\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	fscanf(ptr,"%lf\n",&p->prior_r);
	fgets(line,500,ptr);
	if(strcmp(line,"prior_l\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	fscanf(ptr,"%lf\n",&p->prior_l);
	fgets(line,500,ptr);
	if(strcmp(line,"prior_u\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	fscanf(ptr,"%lf\n",&p->prior_u);
}

void vreadparam(vparameter *p,char *s,FILE *ptr)
{
	char ss[10000];
	double dum[5];
	int dumi;
	
	for(i=1;i<=Number_Simp;i++)
	{
		sprintf(ss,"%s[[%d]]\n",s,i);
		fgets(line,500,ptr);
		printf("\nInput line: %s###",line);
		if(strcmp(line,ss)!=0) {printf("\nFile input error for line=%s. Expected %s###",line,ss); exit(0);}
		fgets(line,500,ptr);
		printf("\nInput line: %s###",line);
		if(strcmp(line,"initial\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
		for(ii=0;ii<5;ii++) fscanf(ptr,"%lf",&dum[ii]);
		fscanf(ptr,"\n");
		p->current[i-1]=dum[repno-1]; // repno is 1..5
		fgets(line,500,ptr);
		printf("\nInput line: %s###",line);
		if(strcmp(line,"propsd\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
		for(ii=0;ii<5;ii++) fscanf(ptr,"%lf",&dum[ii]);
		fscanf(ptr,"\n");
		p->propsd=dum[repno-1]; // repno is 1..5
		fgets(line,500,ptr);
		printf("\nInput line: %s###",line);
		if(strcmp(line,"prior_e\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
		fscanf(ptr,"%d\n",&dumi);
		fgets(line,500,ptr);
		printf("\nInput line: %s###",line);
		if(strcmp(line,"prior_r\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
		fscanf(ptr,"%lf\n",&dum[0]);
		fgets(line,500,ptr);
		printf("\nInput line: %s###",line);
		if(strcmp(line,"prior_l\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
		fscanf(ptr,"%lf\n",&dum[0]);
		fgets(line,500,ptr);
		printf("\nInput line: %s###",line);
		if(strcmp(line,"prior_u\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
		fscanf(ptr,"%lf\n",&dum[0]);
		
	}
}
