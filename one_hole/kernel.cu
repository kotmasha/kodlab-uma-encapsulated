#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "Agent.h"
#include "worker.h"

int worker::t=0;
cudaEvent_t start,stop;
//helper function
/*
*/
__host__ __device__ int compi_GPU(int x){
	if(x%2==0) return x+1;
	else return x-1;
}

__host__ __device__ int ind(int row,int col,int width){
	return row*width+col;
}

__global__ void conjunction_kernel(bool *b1,bool *b2,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		b1[index]=b1[index]&&b2[index];
	}
}

__global__ void disjunction_kernel(bool *b1,bool *b2,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		b1[index]=b1[index]||b2[index];
	}
}


__global__ void negate_disjunction_star_kernel(bool *b1,bool *b2,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		if(index%2==0){
			b1[index]=b1[index]&&!b2[index+1];
		}
		else{
			b1[index]=b1[index]&&!b2[index-1];
		}
	}
}

__global__ void int2bool_kernel(bool *b,int *i,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		if(i[index]==1) b[index]=true;
		else b[index]=false;
	}
}

__global__ void bool2int_kernel(int *i,bool *b,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		if(b[index]) i[index]=1;
		else i[index]=0;
	}
}

__host__ __device__ bool *worker_dir(worker &worker,bool compiY,bool compiX,bool isSymmetry){
	if(!isSymmetry){
		if(!compiY&&!compiX) return worker.dij;
		else if(!compiY&&compiX) return worker.di_j;
		else if(compiY&&!compiX) return worker.d_ij;
		else return worker.d_i_j;
	}
	else{
		if(!compiX&&!compiY) return worker.dji;
		else if(!compiX&&compiY) return worker.dj_i;
		else if(compiX&&!compiY) return worker.d_ji;
		else return worker.d_j_i;
	}
}

__host__ __device__ double *worker_weight(worker &worker,bool compiY,bool compiX){
	if(!compiY&&!compiX) return worker.wij;
	else if(!compiY&&compiX) return worker.wi_j;
	else if(compiY&&!compiX) return worker.w_ij;
	else return worker.w_i_j;
}

//helper function
__device__ bool implies_GPU(worker &worker,bool y,bool x){//implies
	double rc=*(worker_weight(worker,y,x));
	double r_c=*(worker_weight(worker,!y,x));
	double rc_=*(worker_weight(worker,y,!x));
	double r_c_=*(worker_weight(worker,!y,!x));
	double epsilon=(rc+r_c+rc_+r_c_)*worker.threshold;
	double m=min(epsilon,min(rc,min(r_c,r_c_)));
	return rc_<m;
}

__device__ bool equivalent_GPU(worker &worker,bool y,bool x){//equivalent
	double rc=*(worker_weight(worker,y,x));
	double r_c=*(worker_weight(worker,!y,x));
	double rc_=*(worker_weight(worker,y,!x));
	double r_c_=*(worker_weight(worker,!y,!x));
	double epsilon=(rc+r_c+rc_+r_c_)*worker.threshold;
	return rc_==0&&r_c==0;
}

__device__ void orient_square_GPU(worker &worker){//orient_square
	if(worker.sensor_id1==worker.sensor_id2) return;
	*(worker_dir(worker,false,false,false))=false;
	*(worker_dir(worker,false,true,false))=false;
	*(worker_dir(worker,true,false,false))=false;
	*(worker_dir(worker,true,true,false))=false;
	*(worker_dir(worker,false,false,true))=false;
	*(worker_dir(worker,false,true,true))=false;
	*(worker_dir(worker,true,false,true))=false;
	*(worker_dir(worker,true,true,true))=false;

	int square_is_oriented=0;
	for(int x=0;x<2;++x){
		for(int y=0;y<2;++y){
			if(square_is_oriented==0){
				if(implies_GPU(worker,y,x)){
					*(worker_dir(worker,y,x,false))=true;
					*(worker_dir(worker,!y,!x,true))=true;
					*(worker_dir(worker,y,x,true))=false;
					*(worker_dir(worker,!y,!x,false))=false;
					*(worker_dir(worker,!y,x,true))=false;
					*(worker_dir(worker,!y,x,false))=false;
					*(worker_dir(worker,y,!x,false))=false;
					*(worker_dir(worker,y,!x,true))=false;
                    square_is_oriented=1;
				}//implies
				if(equivalent_GPU(worker,y,x)){
					*(worker_dir(worker,y,x,false))=true;
					*(worker_dir(worker,y,x,true))=true;
					*(worker_dir(worker,!y,!x,true))=true;
					*(worker_dir(worker,!y,!x,false))=true;
					*(worker_dir(worker,!y,x,true))=false;
					*(worker_dir(worker,!y,x,false))=false;
					*(worker_dir(worker,y,!x,false))=false;
					*(worker_dir(worker,y,!x,true))=false;
                    square_is_oriented=1;
				}//equivalent
			}//square_is_oriented
		}//j
	}//i
}

__global__ void update_weights_kernel_empirical(worker *workers,bool *observe,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		int y=workers[index].sensor_id1;
		int x=workers[index].sensor_id2;

		*(workers[index].wij)+=observe[2*y]*observe[2*x];
		*(workers[index].w_ij)+=observe[2*y+1]*observe[2*x];
		*(workers[index].wi_j)+=observe[2*y]*observe[2*x+1];
		*(workers[index].w_i_j)+=observe[2*y+1]*observe[2*x+1];
	}
}

__global__ void update_weights_kernel_discounted(worker *workers,bool *observe,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		int y=workers[index].sensor_id1;
		int x=workers[index].sensor_id2;
		double q=workers[index].q;
		*(workers[index].wij)=*(workers[index].wij)*q+(1-q)*observe[2*y]*observe[2*x];
		*(workers[index].w_ij)=*(workers[index].w_ij)*q+(1-q)*observe[2*y+1]*observe[2*x];
		*(workers[index].wi_j)=*(workers[index].wi_j)*q+(1-q)*observe[2*y]*observe[2*x+1];
		*(workers[index].w_i_j)=*(workers[index].w_i_j)*q+(1-q)*observe[2*y+1]*observe[2*x+1];
	}
}

__global__ void calculate_sensor_value(worker *workers,float *sensor_value,int size){//gather all sensor value,workerSize
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		int y=workers[index].sensor_id1;
		int x=workers[index].sensor_id2;
		atomicAdd(sensor_value+2*y,*(workers[index].wij)+*(workers[index].wi_j));
		atomicAdd(sensor_value+2*y+1,*(workers[index].w_ij)+*(workers[index].w_i_j));
		atomicAdd(sensor_value+2*x,*(workers[index].wij)+*(workers[index].w_ij));
		atomicAdd(sensor_value+2*x+1,*(workers[index].wi_j)+*(workers[index].w_i_j));
	}
}

__global__ void update_weights_kernel_distributed(worker *workers,float *sensor_value,bool *observe,int size,int sensorSize,int t){//workerSize
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		int y=workers[index].sensor_id1;
		int x=workers[index].sensor_id2;
		double tij=(*(workers[index].wij)*t+observe[2*y]*observe[2*x])/(t+1);
		double t_ij=(*(workers[index].w_ij)*t+observe[2*y+1]*observe[2*x])/(t+1);
		double ti_j=(*(workers[index].wi_j)*t+observe[2*y]*observe[2*x+1])/(t+1);
		double t_i_j=(*(workers[index].w_i_j)*t+observe[2*y+1]*observe[2*x+1])/(t+1);
		double sij=-*(workers[index].wij)+(sensor_value[2*y]+sensor_value[2*x])/2;
		double s_ij=-*(workers[index].w_ij)+(sensor_value[2*y+1]+sensor_value[2*x])/2;
		double si_j=-*(workers[index].wi_j)+(sensor_value[2*y]+sensor_value[2*x+1])/2;
		double s_i_j=-*(workers[index].w_i_j)+(sensor_value[2*y+1]+sensor_value[2*x+1])/2;

		*(workers[index].wij)=(tij+sij)/(2*sensorSize-3);
		*(workers[index].w_ij)=(t_ij+s_ij)/(2*sensorSize-3);
		*(workers[index].wi_j)=(ti_j+si_j)/(2*sensorSize-3);
		*(workers[index].w_i_j)=(t_i_j+s_i_j)/(2*sensorSize-3);

		//*(workers[index].wij)=tij;
		//*(workers[index].w_ij)=t_ij;
		//*(workers[index].wi_j)=ti_j;
		//*(workers[index].w_i_j)=t_i_j;
	}
}

__global__ void orient_all_kernel(worker *workers,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		orient_square_GPU(workers[index]);
	}
}

__global__ void multiply_kernel(worker *worker,bool *data,bool *affected_worker,int size,int measurableSize){
	int index=threadIdx.x;
	extern __shared__ bool shared[];
	bool *xs=&shared[0];
	bool *ys=&shared[measurableSize];
	__shared__ bool flag[1];
	int x,y;
	int j=index;
	while(j<size){
		x=worker[j].sensor_id2;
		y=worker[j].sensor_id1;
		xs[2*x]=data[2*x];xs[2*x+1]=data[2*x+1];
		xs[2*y]=data[2*y];xs[2*y+1]=data[2*y+1];
		ys[2*x]=data[2*x];ys[2*x+1]=data[2*x+1];
		ys[2*y]=data[2*y];ys[2*y+1]=data[2*y+1];
		j+=512;
	}
	flag[0]=true;
	__syncthreads();
	while(flag[0]){
		flag[0]=false;
		j=index;
		__syncthreads();
		while(j<size){
			x=worker[j].sensor_id2;
			y=worker[j].sensor_id1;
			if(ys[2*x]==0&&xs[2*y]==1&&(*worker[j].dij)) ys[2*x]=1;
			if(ys[2*x+1]==0&&xs[2*y]==1&&(*worker[j].di_j)) ys[2*x+1]=1;
			if(ys[2*x]==0&&xs[2*y+1]==1&&(*worker[j].d_ij)) ys[2*x]=1;
			if(ys[2*x+1]==0&&xs[2*y+1]==1&&(*worker[j].d_i_j)) ys[2*x+1]=1;
			if(ys[2*y]==0&&xs[2*x]==1&&(*worker[j].dji)) ys[2*y]=1;
			if(ys[2*y+1]==0&&xs[2*x]==1&&(*worker[j].dj_i)) ys[2*y+1]=1;
			if(ys[2*y]==0&&xs[2*x+1]==1&&(*worker[j].d_ji)) ys[2*y]=1;
			if(ys[2*y+1]==0&&xs[2*x+1]==1&&(*worker[j].d_j_i)) ys[2*y+1]=1;
			j+=512;
		}
		j=index;
		__syncthreads();
		while(j<size){
			x=worker[j].sensor_id2;
			y=worker[j].sensor_id1;
			if(ys[2*y]==1&&xs[2*y]==0) flag[0]=true;
			if(ys[2*y+1]==1&&xs[2*y+1]==0) flag[0]=true;
			if(ys[2*x]==1&&xs[2*x]==0) flag[0]=true;
			if(ys[2*x+1]==1&&xs[2*x+1]==0) flag[0]=true;
			xs[2*y]=ys[2*y];
			xs[2*y+1]=ys[2*y+1];
			xs[2*x]=ys[2*x];
			xs[2*x+1]=ys[2*x+1];
			j+=512;
		}
		__syncthreads();
	}
	j=index;
	__syncthreads();
	while(j<size){
		x=worker[j].sensor_id2;
		y=worker[j].sensor_id1;
		data[2*x]=ys[2*x];data[2*x+1]=ys[2*x+1];
		data[2*y]=ys[2*y];data[2*y+1]=ys[2*y+1];
		if(affected_worker!=NULL&&ys[2*x]||ys[2*x+1]||ys[2*y]||ys[2*y+1]) affected_worker[j]=true;
		j+=512;
	}
}

//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
__global__ void mask_kernel(bool *mask,int *actionlist,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size){
		for(int i=0;i<size;++i){
			if(index==actionlist[i]){
				mask[index]=true;
				return;
			}
		}
		mask[index]=false;
	}
}

//before invoke this function make sure dev_load and dev_signal have correct data
//the computed data will be in dev_load
void Agent::propagate_GPU(){//propagate
	cudaMemset(dev_affected_worker,0,workerSize*sizeof(bool));

	multiply_kernel<<<1, 512, 2*measurableSize*sizeof(bool)>>>(dev_worker,dev_load,dev_affected_worker,workerSize,measurableSize);

	multiply_kernel<<<1, 512, 2*measurableSize*sizeof(bool)>>>(dev_worker,dev_signal,dev_affected_worker,workerSize,measurableSize);

	// standard operations
	disjunction_kernel<<<(measurableSize+255)/256,256>>>(dev_load,dev_signal,measurableSize);
	negate_disjunction_star_kernel<<<(measurableSize+255)/256,256>>>(dev_load,dev_signal,measurableSize);
	
	cudaMemcpy(Gload,dev_load,measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
	cudaMemcpy(Gaffected_worker,dev_affected_worker,workerSize*sizeof(bool),cudaMemcpyDeviceToHost);
}

void Agent::setSignal(vector<bool> observe){//this is where data comes in in every frame
	for(int i=0;i<observe.size();++i){
		Gobserve[i]=observe[i];
	}
	cudaMemcpy(dev_observe,Gobserve,measurableSize*sizeof(bool),cudaMemcpyHostToDevice);
}

void Agent::update_weights(){}

void Agent_Empirical::update_weights(){
	update_weights_kernel_empirical<<<(workerSize+255)/256,256>>>(dev_worker,dev_observe,workerSize);
}

void Agent_Distributed::update_weights(){
	cudaMemset(dev_sensor_value,0.0,measurableSize*sizeof(float));
	calculate_sensor_value<<<(workerSize+255)/256,256>>>(dev_worker,dev_sensor_value,workerSize);
	update_weights_kernel_distributed<<<(workerSize+255)/256,256>>>(dev_worker,dev_sensor_value,dev_observe,workerSize,sensorSize,worker::t);
}

void Agent_Discounted::update_weights(){
	update_weights_kernel_discounted<<<(workerSize+255)/256,256>>>(dev_worker,dev_observe,workerSize);
}

void Agent::update_state_GPU(bool mode){//true for decide
	float dt=0;
	is_log_on=true;
	if(is_log_on){
		//cudaEventCreate(&start);
		cudaEventRecord(start);
	}
	update_weights();
	n_update_weight++;
	if(is_log_on){
		//cudaEventCreate(&stop);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&dt, start,stop);
		t_update_weight+=dt;
	}
	worker::add_time();//when distributed or multiply agents, need to move it to upper loop
	//update_weight
	
	if(mode){
		if(is_log_on){
			//cudaEventCreate(&start);
			cudaEventRecord(start);
		}
		orient_all_kernel<<<(workerSize+255)/256,256>>>(dev_worker,workerSize);
		n_orient_all++;
		if(is_log_on){
			//cudaEventCreate(&stop);
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&dt, start,stop);
			t_orient_all+=dt;
		}
	}//orient_all

	cudaMemcpy(dev_signal,dev_observe,measurableSize*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load,false,measurableSize*sizeof(bool));
	if(is_log_on){
		//cudaEventCreate(&start);
		cudaEventRecord(start);
	}
	propagate_GPU();
	n_propagation++;
	if(is_log_on){
		//cudaEventCreate(&stop);
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&dt, start,stop);
		t_propagation+=dt;
	}
	cudaMemcpy(Gcurrent,dev_load,measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_current,dev_load,measurableSize*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemcpy(Gdir,dev_dir,measurableSize*measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
}

void Agent::halucinate_GPU(vector<int> actions_list){
	//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
	vector<bool> mask=initMask(actions_list);
	vector<int> v;
	for(int i=0;i<actions_list.size();++i){
		for(int j=0;j<measurableSize;++j){
			if(context.find(pair<int,int>(actions_list[i],j))!=context.end()&&Gcurrent[j]){
				v.push_back(context[pair<int,int>(actions_list[i],j)]);
			}
		}
	}
	//relevant_pairs=[(act,ind) for act in actions_list for ind in xrange(self._SIZE) if (act,ind) in self._CONTEXT and self._CURRENT.value(ind)]
	//map(mask.set,[self._CONTEXT[i,j] for i,j in relevant_pairs],[True for i,j in relevant_pairs])
	for(int i=0;i<v.size();++i) mask[v[i]]=true;
	
	for(int i=0;i<mask.size();++i){
		Gmask[i]=mask[i];
	}
	cudaMemcpy(dev_mask,Gmask,measurableSize*sizeof(bool),cudaMemcpyHostToDevice);
	//copy data
	cudaMemcpy(dev_signal,dev_mask,measurableSize*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_load,dev_current,measurableSize*sizeof(bool),cudaMemcpyDeviceToDevice);
	propagate_GPU();
	//return self.propagate(mask,self._CURRENT)
}

void Agent::freeData(){//free data in case of memory leak
	delete[] Gdir;
	delete[] Gweights;
	delete[] Gthresholds;
	delete[] Gobserve;
	delete[] Gdfs;
	delete[] Gsignal;
	delete[] Gload;

	delete[] Gmask;
	delete[] Gcurrent;
	delete[] Gworker;
	delete[] Gaffected_worker;

	cudaFree(dev_worker);
	cudaFree(dev_dir);
	cudaFree(dev_thresholds);
	cudaFree(dev_weights);
	cudaFree(dev_observe);
	cudaFree(dev_dfs);
	cudaFree(dev_signal);
	cudaFree(dev_load);
	
	cudaFree(out_load);
	cudaFree(out_signal);

	cudaFree(dev_scan);

	cudaFree(dev_mask);
	cudaFree(dev_current);

	cudaFree(dev_sensor_value);

	cudaFree(dev_affected_worker);
}

void Agent::initData(string name,int sensorSize,vector<vector<int> > context_key,vector<int> context_value,
		vector<string> sensors_names,vector<string> evals_names,vector<vector<int> > generalized_actions){
	//data init
	this->name=name;
	this->sensorSize=sensorSize;
	this->measurableSize=2*sensorSize;
	this->workerSize=sensorSize*(sensorSize-1)/2;
	this->sensors_names=sensors_names;
	this->evals_names=evals_names;
	this->generalized_actions=generalized_actions;
	srand (time(NULL));
	for(int i=0;i<measurableSize;++i){
		name_to_num[sensors_names[i]]=i;
	}
	if(Gdir!=NULL){
		freeData();
	}
	
	Gdir=new bool[measurableSize*measurableSize];
	Gweights=new double[measurableSize*measurableSize];
	Gthresholds=new double[measurableSize*measurableSize];
	Gobserve=new bool[measurableSize];
	Gdfs=new bool[1];
	Gsignal=new bool[measurableSize];
	Gload=new bool[measurableSize];

	Gmask=new bool[measurableSize];
	Gcurrent=new bool[measurableSize];
	Gworker=new worker[workerSize];
	Gaffected_worker=new bool[workerSize];
	
	cudaMalloc(&dev_dir,measurableSize*measurableSize*sizeof(bool));
	cudaMalloc(&dev_thresholds,measurableSize*measurableSize*sizeof(double));
	cudaMalloc(&dev_weights,measurableSize*measurableSize*sizeof(double));
	cudaMalloc(&dev_observe,measurableSize*sizeof(bool));
	cudaMalloc(&dev_dfs,sizeof(bool));
	cudaMalloc(&dev_signal,measurableSize*sizeof(bool));
	cudaMalloc(&dev_load,measurableSize*sizeof(bool));
	cudaMalloc(&dev_sensor_value,measurableSize*sizeof(float));

	initWorkerMemory(dev_weights,dev_dir);
	cudaMalloc(&dev_worker,workerSize*sizeof(worker));
	cudaMemcpy(dev_worker,Gworker,workerSize*sizeof(worker),cudaMemcpyHostToDevice);

	cudaMalloc(&out_signal, measurableSize*sizeof(int));
	cudaMalloc(&out_load, measurableSize*sizeof(int));

	cudaMalloc(&dev_mask,measurableSize*sizeof(bool));
	cudaMalloc(&dev_current,measurableSize*sizeof(bool));

	cudaMalloc(&dev_scan,measurableSize*sizeof(int));

	cudaMalloc(&dev_affected_worker,workerSize*sizeof(bool));

	// we need to make diagonals of Gdir = 1, true
	for(int i=0;i<measurableSize;++i){
		for(int j=0;j<measurableSize;++j){
			Gthresholds[i*measurableSize+j]=threshold;
			Gweights[i*measurableSize+j]=0.0;
			Gdir[i*measurableSize+j]=false;
			// new implementation
			if(i == j)
				Gdir[i*measurableSize+j] = true;
		}
	}

	cudaMemcpy(dev_thresholds,Gthresholds,measurableSize*measurableSize*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weights,Gweights,measurableSize*measurableSize*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dir,Gdir,measurableSize*measurableSize*sizeof(bool),cudaMemcpyHostToDevice);
	//init threshold

	for(int i=0;i<context_key.size();++i){
		context[pair<int,int>(context_key[i][0],context_key[i][1])]=context_value[i];
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cout<<"succeed"<<endl;
}

void Agent::initWorkerMemory(double *weights,bool *dir){
	int y=0,x=y+1;
	for(int i=0;i<workerSize;++i){
		Gworker[i]=worker("","",y,x);//name input required
		Gworker[i].wij=&weights[ind(2*y,2*x,measurableSize)];
		Gworker[i].w_ij=&weights[ind(2*y+1,2*x,measurableSize)];
		Gworker[i].wi_j=&weights[ind(2*y,2*x+1,measurableSize)];
		Gworker[i].w_i_j=&weights[ind(2*y+1,2*x+1,measurableSize)];

		Gworker[i].dij=&dir[ind(2*y,2*x,measurableSize)];
		Gworker[i].d_ij=&dir[ind(2*y+1,2*x,measurableSize)];
		Gworker[i].di_j=&dir[ind(2*y,2*x+1,measurableSize)];
		Gworker[i].d_i_j=&dir[ind(2*y+1,2*x+1,measurableSize)];

		Gworker[i].dji=&dir[ind(2*x,2*y,measurableSize)];
		Gworker[i].d_ji=&dir[ind(2*x+1,2*y,measurableSize)];
		Gworker[i].dj_i=&dir[ind(2*x,2*y+1,measurableSize)];
		Gworker[i].d_j_i=&dir[ind(2*x+1,2*y+1,measurableSize)];

		x++;
		if(x==sensorSize){
			y++;
			x=y+1;
		}
		Gworker[i].threshold=threshold;
	}
}

void Agent_Empirical::initWorkerMemory(double *weights,bool *dir){
	Agent::initWorkerMemory(weights,dir);
}

void Agent_Distributed::initWorkerMemory(double *weights,bool *dir){
	Agent::initWorkerMemory(weights,dir);
}

void Agent_Discounted::initWorkerMemory(double *weights,bool *dir){
	Agent::initWorkerMemory(weights,dir);
	for(int i=0;i<workerSize;++i){
		Gworker[i].q=q;
	}
}