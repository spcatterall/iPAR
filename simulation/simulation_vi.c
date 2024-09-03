// Simulation of the iPAR model
// Stephen Catterall

// Assumes that the background infection rate depends ONLY on the eps parameters and the kernel parameter lambda
// Assumes that Number_Eps is (at least) 6
// NB To run on Linux, rename simulation e.g. input file as siminputsw.txt, then use tr -d '\r' <siminputsw.txt> siminputs.txt 

#include <stdio.h>
#include <math.h>
#include <time.h>   
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
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
#define MSI 100 // maximum number of time subintervals (for generating risk maps)
#define NUMR 10 // maximum number of regions into which the study area is split for the 'risktemporal' output
#define TIMEINDEXMAX 10000 // maximum number of iterations in a simulation
#define MAXSAMPLES 500000 // maximum number of MCMC samples to be read in (usually this will be 450000)

struct unitss{
  double x;
  double y;
  double h[MNCOV]; // simplex covariates
  double k[MNCOV]; // exponential covariates
  double ep[MNCOV]; // epsilon covariates: pest,plat,plit,da,db,dc or whatever
  double eps; // value of (background infection rate)/(suitability) for this unit
  int    initial;
  double susceptibility;
  double infectivity;
  double infection_time;
  double risk[MSI]; // for risk maps
  int 	 region[NUMR+1]; // for risktemporal output
  double foi; // force of infection 
};
typedef struct unitss units;

struct next_ev{ // next event characteristics stored in a realisation of this data structure: only applies to event categories for which events can occur at any time and have an associated rate
  int type; // 1=infection
  int unit1; // first unit to which event applies (for infection, the unit that becomes infected)
  int unit2; // second unit to which event applies (for infection, the infector unit)
};
typedef struct next_ev next_even;

struct listtime{ // store details of next 'deterministic' event i.e. those events excluded from the next_ev structure above
  int type; // 1=changepoint
  int unit; // unit associated with event
  double time; // time at which event is scheduled to occur
  int timeindex; // index of time changepoint (0...ntk-1)
};
typedef struct listtime listime;

int 	  nb[MNU][NBMAX]={{0}}; // list of N(i)\I(i) for each unit i
int 	  nbinfec[MNU][NBMAX]={{0}}; // list of N(i)^I(i) for each unit i
int 	  infec[MNU][NBMAX]={{0}}; // list of I(i)\N(i) for each unit i
int 	  nbsize[MNU]={0}; // size of nb for each unit (should always be <=NBMAX)
int 	  nbinfecsize[MNU]={0};
int 	  infecsize[MNU]={0};
double    linknb[MNU][NBMAX]={{0.0}}; // matrix of link strengths between interacting units (ignoring susceptibility and infectivity factors), with entries mirroring those in matrix nb
double    linknbinfec[MNU][NBMAX]={{0.0}}; // NB linknbinfec[i][j]:= distance dependent transmission rate from nbinfec[i][j] to i
double    revlinknbinfec[MNU][NBMAX]={{0.0}}; // NB revlinknbinfec[i][j]:= distance dependent transmission rate from i to nbinfec[i][j] 
double    linkinfec[MNU][NBMAX]={{0.0}}; // linkinfec[i][j]:= distance dependent transmission rate from i to infec[i][j]
listime lis={0};
next_even next_event;
double (*epscalc) (int); // function pointer for the epscalc function
double lambda,rho,b,timenow,totalrate,dt,z,sum,TEND,verysmall,dummy,multiplier,exponent;
double subinttimes[1000];
double sigma[MNCOV];
double sigmap[MNCOV];
double gama[MNCOV];
double gamap[MNCOV];
double eps[MNCOV];
double f[MNCP];
const gsl_rng_type *T;
gsl_rng *r;
FILE *fptr;
FILE *cfptr;
char fname[100];
char sdummy[100];
char line[10000];
double tk[1000];
int iobs[MNU];
int      i,i2,j,k,k2,k3,kk,s,ninitial,nsampf,nf,number_atlases,nsim,sample,nothing_happens,nsubints,nunits,nregions,x_distance,y_distance,timeindex,dx,dy,seed,numtk,timeclass,ecount,filestart,idummy,i1,Number_Simp,Number_Eps,Number_Exp,model,burnin,dmax;
units unit[MNU];
double    kernel_matrix[DMAX+1][DMAX+1]={{0.0}};// kernel_matrix[x][y] stores baseline secondary transmission rate at relative coordinates [x][y] from source of infection
double sigmat[MAXSAMPLES][MNCOV]={{0}}; // samples of parameters
double gamat[MAXSAMPLES][MNCOV]={{0}};
double sigmapt[MAXSAMPLES][MNCOV]={{0}}; 
double gamapt[MAXSAMPLES][MNCOV]={{0}};
double rhot[MAXSAMPLES]={0};
double ft[MAXSAMPLES][MNCP]={{0}};
double lambdat[MAXSAMPLES]={0};
double epsilont[MAXSAMPLES][MNCOV]={{0}};

double risktemporal[NUMR+1][MSI+1][MNU+1]={{0}};
double incidence[NUMR+1][MSI+1][MNU+1]={{0}};
double riskspatial[MSI+1][MNU]={{0}};

double kernel(int,int,double);
double epscalcfull(int);
double epscalcmini(int);
int fabby(double);
double evalf(double,double *);
void read_inputs(void);
void readint(int *,char *,FILE *);

double epscalcfull(int i) // compute background infection rate for unit i (full model)
{
	double out;
	out=unit[i].ep[0]*eps[0]+unit[i].ep[1]*eps[1]+unit[i].ep[2]*eps[2]+pow(unit[i].ep[3],-2.0*lambda)*eps[3]+pow(unit[i].ep[4],-2.0*lambda)*eps[4]+pow(unit[i].ep[5],-2.0*lambda)*eps[5];
	return(out);
}

double epscalcmini(int i) // compute background infection rate for unit i (minimal model)
{
	double out,min;
	
	min=unit[i].ep[5];
	if(unit[i].ep[3]<min) min=unit[i].ep[3];
	if(unit[i].ep[4]<min) min=unit[i].ep[4];
	out=(unit[i].ep[0]+unit[i].ep[1]+unit[i].ep[2])*eps[0]+pow(min,-2.0*lambda)*eps[3];
	return(out);
}

int main()
{
  // Main outputs:: riskspatial: r1...rNH (time point 1), r1...rNH (time point 2) etc.  risktemporal: P(0 infected)...P(nunits infected) (@ time point 1), P(0 infected)...P(nunits infected) (@ time point 2), etc. up to TEND (=final time point)
  
  // Initialise GSL random number generator and miscellaneous other things
  gsl_rng_env_setup();
  T=gsl_rng_default;
  r = gsl_rng_alloc(T); 
  seed=(int)(time(NULL)); // Set random seed using computer clock
  gsl_rng_set(r,seed); 
  
  // Initialise units
  for(i=0;i<MNU;i++)
    {
	  unit[i].x=0;
	  unit[i].y=0;
	  for(k=0;k<MNCOV;++k)unit[i].h[k]=0.0;
	  for(k=0;k<MNCOV;++k)unit[i].k[k]=0.0;
	  for(k=0;k<MNCOV;k++) unit[i].ep[k]=0.0;
	  unit[i].eps=0; 
	  unit[i].infection_time=TEND+1; // this means that the infection time is not yet fixed for this unit
	  unit[i].susceptibility=0;
	  unit[i].infectivity=0;
	  unit[i].initial=0;
	  for(k=0;k<MSI;k++) unit[i].risk[k]=0.0;
	  for(k=0;k<NUMR;k++) unit[i].region[k]=0;
	  unit[i].foi=0;
    }
  
  // Read in inputs
  read_inputs();
  for(i=0;i<nunits;i++) unit[i].region[nregions]=1; // this defines a standard region which includes ALL units
  if(nunits>MNU) {printf("\nError!!! MNU<nunits..."); exit(1);}
  if(nregions>NUMR) {printf("\nError!!! NUMR<nregions..."); exit(1);} 
  verysmall=exp(-100.0);

  if(numtk>MNCP) {printf("\nError!!! Inconsistency in the number of check points..."); exit(1);}
  printf("\nNumber of units is %d which is less than %d",nunits,MNU);
  for(i=0;i<nunits;i++) {nbsize[i]=0; nbinfecsize[i]=0; infecsize[i]=0;} // very important this bit! 
  if(model==0) epscalc=&epscalcmini;
  if(model==1) epscalc=&epscalcfull;
  
  TEND=0.0;
  for(k=0;k<=nsubints;k++)
  {
	if(subinttimes[k]>TEND) TEND=subinttimes[k];
  }
  printf("\nTEND=%f",TEND);

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

  // Read in initial state
  cfptr=fopen("init.dat","r");
  fscanf(cfptr,"%d",&ninitial);
  for(k=0;k<ninitial;k++) fscanf(cfptr,"%d",&iobs[k]);
  fclose(cfptr);
  
  // Add initial map to infectious unit data structure
  for(k=0;k<ninitial;k++)
    {
      unit[iobs[k]].initial=1;
	  unit[iobs[k]].infection_time=0;
    }
  
  // Define neighbourhoods (same framework as developed for the inference algorithm)
  for(i=0;i<nunits;i++)
  {
	  // Compute nbhd for unit i   
	  for(j=0;j<nunits;j++)
	  {
		  if(kernel(fabby(unit[i].x-unit[j].x),fabby(unit[i].y-unit[j].y),2)>0) {nbinfec[i][nbinfecsize[i]]=j; nbinfecsize[i]++;} // lambda=2 arbitrary
		  if(nbinfecsize[i]>NBMAX-2) {printf("\nError: max nhbd size exceeded!"); exit(0);}
	  }
  }
  
  // Read in parameter samples/values
  filestart=1;
  if((nf==1)&&(nsampf==1)) filestart=0; // if generating a simulated dataset (rather than a posterior predictive distribution) input file is par_0 rather than par_1..5
  for(i=filestart;i<filestart+nf;i++) // loop over input files
  {
	  i1=i-filestart;
	  sprintf(fname,"par_%d.txt",i); 
	  fptr=fopen(fname,"r");
	  for(j=0;j<nsampf;j++) // loop through lines of the file
	  {
		  fscanf(fptr,"%lf",&lambdat[i1*nsampf+j]);
		  fscanf(fptr,"%lf",&rhot[i1*nsampf+j]);
		  for(k=0;k<Number_Eps;k++) fscanf(fptr,"%lf",&epsilont[i1*nsampf+j][k]);
		  for(k=0;k<Number_Simp;k++) fscanf(fptr,"%lf",&sigmat[i1*nsampf+j][k]);
		  for(k=0;k<Number_Simp;k++) fscanf(fptr,"%lf",&gamat[i1*nsampf+j][k]);
		  for(k=0;k<numtk;k++) fscanf(fptr,"%lf",&ft[i1*nsampf+j][k]);
		  for(k=0;k<Number_Exp;k++) fscanf(fptr,"%lf",&sigmapt[i1*nsampf+j][k]);
		  for(k=0;k<Number_Exp;k++) fscanf(fptr,"%lf",&gamapt[i1*nsampf+j][k]);
		  fscanf(fptr,"%lf",&dummy); // loglikelihood value
		  for(k=0;k<Number_Simp-1;k++) fscanf(fptr,"%lf",&dummy);
		  for(k=0;k<Number_Simp-1;k++) fscanf(fptr,"%lf",&dummy);
	  }
	  fclose(fptr);
  }

  // Simulation
  for(s=0;s<nsim;s++)
    {
	  ecount=0;
	  // Sample model parameters
	  printf("\ns=%d",s);
	  sample=nsampf*gsl_rng_uniform_int(r,nf)+burnin+gsl_rng_uniform_int(r,nsampf-burnin);
	  
	  lambda=lambdat[sample];
	  rho=rhot[sample];
	  for(k2=0;k2<Number_Eps;k2++)eps[k2]=epsilont[sample][k2];
	  for(k2=0;k2<Number_Simp;k2++)sigma[k2]=sigmat[sample][k2];
	  for(k2=0;k2<Number_Simp;k2++)gama[k2]=gamat[sample][k2];
	  for(k2=0;k2<numtk;k2++)f[k2]=ft[sample][k2];
	  for(k2=0;k2<Number_Exp;k2++)sigmap[k2]=sigmapt[sample][k2];
	  for(k2=0;k2<Number_Exp;k2++)gamap[k2]=gamapt[sample][k2];
	  
	  // Compute transmission kernel matrix
      b=0.0;
      for(x_distance=0;x_distance<=dmax;++x_distance)
      {
        for(y_distance=0;y_distance<=dmax;++y_distance)
	    {
	      kernel_matrix[x_distance][y_distance]=kernel(x_distance,y_distance,lambda);
	      if((x_distance==0)&&(y_distance!=0))b=b+2.0*kernel_matrix[x_distance][y_distance];
	      if((x_distance!=0)&&(y_distance==0))b=b+2.0*kernel_matrix[x_distance][y_distance];
	      if((x_distance!=0)&&(y_distance!=0))b=b+4.0*kernel_matrix[x_distance][y_distance];
	    }
      }
      for(x_distance=0;x_distance<=dmax;x_distance++){for(y_distance=0;y_distance<=dmax;y_distance++)kernel_matrix[x_distance][y_distance]=kernel_matrix[x_distance][y_distance]/b;}//normalises the kernel_matrix
	
	  // Compute link strengths, suitabilities & background transmission rates
      for(i=0;i<nunits;i++) 
	  {
		for(j=0;j<nbsize[i];j++) linknb[i][j]=kernel_matrix[fabby(unit[i].x-unit[nb[i][j]].x)][fabby(unit[i].y-unit[nb[i][j]].y)];
	    for(j=0;j<nbinfecsize[i];j++) {linknbinfec[i][j]=kernel_matrix[fabby(unit[i].x-unit[nbinfec[i][j]].x)][fabby(unit[i].y-unit[nbinfec[i][j]].y)]; revlinknbinfec[i][j]=linknbinfec[i][j];}
	    for(j=0;j<infecsize[i];j++) linkinfec[i][j]=kernel_matrix[fabby(unit[i].x-unit[infec[i][j]].x)][fabby(unit[i].y-unit[infec[i][j]].y)];
		unit[i].susceptibility=0;
		for(k=0;k<Number_Simp;++k) unit[i].susceptibility += sigma[k]*unit[i].h[k];
		exponent=0;
		for(k=0;k<Number_Exp;k++) exponent+=sigmap[k]*unit[i].k[k];
		unit[i].susceptibility*=exp(exponent);
		unit[i].infectivity=0; 
		for(k=0;k<Number_Simp;++k) unit[i].infectivity += gama[k]*unit[i].h[k];
		exponent=0;
		for(k=0;k<Number_Exp;k++) exponent+=gamap[k]*unit[i].k[k];
		unit[i].infectivity*=exp(exponent);
		unit[i].eps=epscalc(i);
		
      }
      
	  // Perform simulation
      timenow=0;
	  timeindex=0;
	  totalrate=0;
	  // Initialise infection times
	  for(i=0;i<nunits;i++)
	  {
		  unit[i].infection_time=TEND+1;
	  }
	  for(k=0;k<ninitial;k++)
      {
	    unit[iobs[k]].infection_time=0;
      }
	  // Compute initial total event rate and FOIs
	  // 1. Primary infection
	  for(i=0;i<nunits;i++)
	  {
		  if((unit[i].susceptibility>0)&&(unit[i].infection_time>TEND)) // if not initially infected...
		  {
			  unit[i].foi=unit[i].eps*unit[i].susceptibility;
			  totalrate+=unit[i].foi;
			  //printf("\nfoi=%f",unit[i].foi);
		  }
	  }
	  //printf("\nPrimary=%f",totalrate);
	  
	  // 2. Secondary infection
	  for(i=0;i<nunits;i++) // Loop over all units i...
	  {
		  if((unit[i].infectivity>0)&&(unit[i].infection_time<=timenow)) // ...  that are currently sources of infection
		  {
			  for(j=0;j<nbinfecsize[i];j++) // now loop over all j within the neighbourbood of i...
			  {
				  if((unit[nbinfec[i][j]].susceptibility>0) && (unit[nbinfec[i][j]].infection_time>TEND)) // that can potentially be infected by i
				  {
					  totalrate+=revlinknbinfec[i][j]*unit[i].infectivity*rho*unit[nbinfec[i][j]].susceptibility; // NB f=1 in the first time segment so is omitted from the expression here!
					  unit[nbinfec[i][j]].foi+=revlinknbinfec[i][j]*unit[i].infectivity*rho*unit[nbinfec[i][j]].susceptibility;
				  }
			  }
		  }
	  }
	  //printf("\nSecondary+primary=%f",totalrate);
	  
	  lis.time=1000000;
	  for(i=0;i<numtk;i++)
	  {
		  if((tk[i]<lis.time)&&(tk[i]>timenow)) {lis.time=tk[i];lis.timeindex=i;}
	  }
	  
	  //dt=-log(gsl_rng_uniform(r))/totalrate; 
	  //timenow+=dt;
	 
	  while((timenow<TEND)&&(timeindex<TIMEINDEXMAX)) // while time is still less than the final time point TEND
	  {
		  if(totalrate>0) 
		  {
			  dt=-log(gsl_rng_uniform(r))/totalrate; 
			  if(lis.time<timenow+dt) // it's a deterministic event
			  {
				  timenow=lis.time;
				  // Adjust rates of infection
				  if(lis.timeindex==0) multiplier=f[0];
				  if(lis.timeindex>0) multiplier=f[lis.timeindex]/f[lis.timeindex-1];
				  for(i=0;i<nunits;i++) unit[i].foi*=multiplier;
				  totalrate*=multiplier;
				  // Recompute lis.time
				  lis.time=1000000;
	              for(i=0;i<numtk;i++)
	              {
		            if((tk[i]<lis.time)&&(tk[i]>timenow)) {lis.time=tk[i];lis.timeindex=i;}
	              }
			  }
			  else // it's a rate-based event so determine which one and then implement it
			  {
				ecount++;
				// Determine the nature of the next event
			    //printf("\nTotalrate=%f",totalrate);
		        z=gsl_rng_uniform(r);
			    sum=0;
		        next_event.type=0;
		        next_event.unit1=-1;
		        next_event.unit2=-1;
		        for(i=0;i<nunits;i++)
	            {
		          if((unit[i].susceptibility>0)&&(unit[i].infection_time>TEND))
		          {
			        sum+=unit[i].foi;
			        if((z<sum/totalrate) && (next_event.type==0)) {next_event.type=1; next_event.unit1=i;}
		          }
	            }
		        if(next_event.type==0) {printf("\nError! Next event has not been determined!\n"); exit(0);}
		  
		        // Implement next event
		        timeindex++;
		  
		        // Recompute total rate and FOIs
				timenow+=dt;
				unit[next_event.unit1].infection_time=timenow;
		        i=next_event.unit1;
		        totalrate-=unit[i].foi; // i now infected so remove its FOI from the total rate
		        for(j=0;j<nbinfecsize[i];j++)
		        {
			        if((unit[nbinfec[i][j]].susceptibility>0) && (unit[nbinfec[i][j]].infection_time>TEND))
			        {
				       totalrate+=evalf(timenow,tk)*revlinknbinfec[i][j]*unit[i].infectivity*rho*unit[nbinfec[i][j]].susceptibility;
				       unit[nbinfec[i][j]].foi+=evalf(timenow,tk)*revlinknbinfec[i][j]*unit[i].infectivity*rho*unit[nbinfec[i][j]].susceptibility;
			        }
		        }
			  }
		  }
		  else // event rate is zero so either look for deterministic events or increment time to end of the simulation (as in SIC model)
		  {
			  timenow++; // if there are deterministic events these are just multiplying zero FOIs, so it's pointless to implement these events; therefore just increment time
		  }
	  }
	  
	  // Calculate infection probabilities
      for(i=0;i<nunits;i++)
	  {
		for(k=0;k<=nsubints;k++)
		{
		  if(unit[i].infection_time<=subinttimes[k]) unit[i].risk[k]+=1.0/nsim;
		}	
	  }
      
      // Make temporal output for each region  
      for(k=0;k<=nsubints;k++)
	  {
		for(kk=0;kk<=nregions;kk++)
		{
	      k2=0;
		  k3=0;
	      for(i=0;i<nunits;i++)
	      {
		    if((unit[i].region[kk])&&(unit[i].infection_time<=subinttimes[k]))k2++;   
			if(k>0) {if((unit[i].region[kk])&&(unit[i].infection_time<=subinttimes[k])&&(unit[i].infection_time>subinttimes[k-1]))k3++;}
			if(k==0) {if((unit[i].region[kk])&&(unit[i].infection_time<=subinttimes[k]))k3++;}
		  }
		  
	      risktemporal[kk][k][k2]+=1.0/nsim;
		  incidence[kk][k][k3]+=1.0/nsim;
		}
	  }
	  
	  if((nf==1)&&(nsampf==1)) // 1 parameter set to observation group
	  {
		  sprintf(fname,"obsg_%d.dat",s);
		  fptr=fopen(fname,"w");
		  
		  for(i=0;i<nunits;i++)
		  {
			  timeclass=1000;
			  if(unit[i].initial) timeclass=0;
			  for(k=1;k<=nsubints;k++) {if((unit[i].infection_time>subinttimes[k-1])&&(unit[i].infection_time<=subinttimes[k])) timeclass=k;}
			  fprintf(fptr,"%d %d 1 %d %f\n",(int)unit[i].x,(int)unit[i].y,timeclass,unit[i].infection_time);
		  }
		  fclose(fptr);
	  }
	  printf("\nEcount=%d",ecount);

    }//end simulation s

	// Pull risk maps together
    for(i=0;i<nunits;i++)
    {
	  for(k=0;k<=nsubints;k++) riskspatial[k][i]=unit[i].risk[k];
	} 
	
	// Write outputs to files
	fptr=fopen("riskspatial.txt","w");
	for(i=0;i<nunits;i++)
    {
	  for(k=0;k<=nsubints;k++) fprintf(fptr,"%f ",riskspatial[k][i]);
	  fprintf(fptr,"\n");
	}
	fclose(fptr);
	
	fptr=fopen("risktemporal.txt","w");
	cfptr=fopen("incidence.txt","w");
	for(kk=0;kk<=nregions;kk++) // loop over regions (always at least one , the standard region which covers everything)
    {
	  for(i=0;i<=nunits;i++)
	  {
		  for(k=0;k<=nsubints;k++) fprintf(fptr,"%f ",risktemporal[kk][k][i]);
		  for(k=0;k<=nsubints;k++) fprintf(cfptr,"%f ",incidence[kk][k][i]);
		  fprintf(fptr,"\n");
		  fprintf(cfptr,"\n");
	  }
	}
	fclose(fptr);
	fclose(cfptr);

}


double kernel(int i, int j, double lam)//returns kernel rate from (0,0) to (i,j)
{
  double output=0.0, distance_squared;
  distance_squared = (double)(i*i) + (double)(j*j);
  if(distance_squared>0) output = pow(distance_squared,-1.0*lam);//ie rate is d^{-2lambda}
  if(distance_squared>((double)dmax)*((double)dmax)) output = 0.0;
  return(output);
}

int fabby(double x) // returns absolute value of x as an integer
{
	int zc;
	
	zc=(int)(x);
	return(abs(zc));
}

double evalf(double u,double *tk) // evaluate time varying function f(t) at a specific t=u
{
	double out=1;
	int k;
	
	for(k=0;k<numtk;k++) {if(u>tk[k]) out=f[k];}
	
	return(out);
}

void read_inputs(void)
{
	FILE *ptr;
	double dum[5];
	char pname[100];
	
	ptr=fopen("inputs.txt","r");
	
	readint(&Number_Simp,"Number_Simp",ptr);
	readint(&Number_Eps,"Number_Eps",ptr);
	readint(&Number_Exp,"Number_Exp",ptr);
	readint(&nunits,"nunits",ptr);
	readint(&nsubints,"nsi",ptr);
		
	fgets(line,500,ptr);
	if(strcmp(line,"subinttimes\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<=nsubints;i++) fscanf(ptr,"%lf",&subinttimes[i]);
	fscanf(ptr,"\n");
	
	readint(&model,"model",ptr);
	readint(&numtk,"numtk",ptr);
	if(numtk>1000) {printf("\nError! tk too big."); exit(0);}
	
	fgets(line,500,ptr);
	if(strcmp(line,"tk\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<=numtk;i++) fscanf(ptr,"%lf",&tk[i]);
	fscanf(ptr,"\n");
	
	readint(&dmax,"maxdisp",ptr);
	readint(&nregions,"numregions",ptr);
	
	fgets(line,500,ptr);
	if(strcmp(line,"regions\n")!=0) {printf("\nFile input error for line=%s.",line); exit(0);}
	for(i=0;i<nregions;i++)
	{
		for(j=0;j<nunits;j++) fscanf(ptr,"%d",&unit[j].region[i]);
	}	
	fscanf(ptr,"\n");
	
	readint(&nf,"numfiles",ptr);
	readint(&nsampf,"numsamplesperfile",ptr);
	readint(&burnin,"burnin",ptr);
	readint(&nsim,"numsims",ptr);
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



