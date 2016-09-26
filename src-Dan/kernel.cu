#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "Agent.h"
#include "worker.h"
#include "logging.h"

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

/*
------------------------------worker-kernel------------------------------
*/

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

/*
------------------------------worker-kernel------------------------------
*/


/*
-----------------------------------------non-worker-kernel-----------------------------------
*/


__device__ bool implies_GPU(int row,int col,int width,double *weights,double threshold){//implies
	double rc=weights[ind(row,col,width)];
	double r_c=weights[ind(compi_GPU(row),col,width)];
	double rc_=weights[ind(row,compi_GPU(col),width)];
	double r_c_=weights[ind(compi_GPU(row),compi_GPU(col),width)];
	double epsilon=(rc+r_c+rc_+r_c_)*threshold;
	double m=min(epsilon,min(rc,min(r_c,r_c_)));
	return rc_<m;
}

__device__ bool equivalent_GPU(int row,int col,int width,double *weights,double threshold){//equivalent
	double rc=weights[ind(row,col,width)];
	double r_c=weights[ind(compi_GPU(row),col,width)];
	double rc_=weights[ind(row,compi_GPU(col),width)];
	double r_c_=weights[ind(compi_GPU(row),compi_GPU(col),width)];
	double epsilon=(rc+r_c+rc_+r_c_)*threshold;
	return rc_==0&&r_c==0;
}

__device__ void orient_square_GPU(bool *dir,double *weights,double *thresholds,int x,int y,int width){//orient_square
	dir[ind(x,y,width)]=false;
	dir[ind(x,compi_GPU(y),width)]=false;
	dir[ind(compi_GPU(x),y,width)]=false;
	dir[ind(compi_GPU(x),compi_GPU(y),width)]=false;
	dir[ind(y,x,width)]=false;
	dir[ind(compi_GPU(y),x,width)]=false;
	dir[ind(y,compi_GPU(x),width)]=false;
	dir[ind(compi_GPU(y),compi_GPU(x),width)]=false;

	int square_is_oriented=0;
	for(int i=0;i<2;++i){
		for(int j=0;j<2;++j){
			int sx=x+i;
            int sy=y+j;
			if(square_is_oriented==0){
				if(implies_GPU(sy,sx,width,weights,thresholds[ind(sy,sx,width)])){
					dir[ind(sy,sx,width)]=true;
					dir[ind(compi_GPU(sx),compi_GPU(sy),width)]=true;
					dir[ind(sx,sy,width)]=false;
                    dir[ind(compi_GPU(sy),compi_GPU(sx),width)]=false;
                    dir[ind(sx,compi_GPU(sy),width)]=false;
                    dir[ind(compi_GPU(sy),sx,width)]=false;
                    dir[ind(sy,compi_GPU(sx),width)]=false;
                    dir[ind(compi_GPU(sx),sy,width)]=false;
                    square_is_oriented=1;
				}//implies
				if(equivalent_GPU(sy,sx,width,weights,thresholds[ind(sy,sx,width)])){
					dir[ind(sy,sx,width)]=true;
					dir[ind(sx,sy,width)]=true;
					dir[ind(compi_GPU(sx),compi_GPU(sy),width)]=true;
                    dir[ind(compi_GPU(sy),compi_GPU(sx),width)]=true;
					dir[ind(sx,compi_GPU(sy),width)]=false;
                    dir[ind(compi_GPU(sy),sx,width)]=false;
                    dir[ind(sy,compi_GPU(sx),width)]=false;
                    dir[ind(compi_GPU(sx),sy,width)]=false;
                    square_is_oriented=1;
				}//equivalent
			}//square_is_oriented
		}//j
	}//i
}

__global__ void update_weights_kernel(double *weights,bool *observe,int size){
	int indexX=blockDim.x*blockIdx.x+threadIdx.x;
	int indexY=blockDim.y*blockIdx.y+threadIdx.y;
	if(indexX<size&&indexY<size){
		weights[ind(indexY,indexX,size)]+=observe[indexX]*observe[indexY];
	}
}

__global__ void orient_all_kernel(bool *dir,double *weights,double *thresholds,int size){
	int indexX=blockDim.x*blockIdx.x+threadIdx.x;
	int indexY=blockDim.y*blockIdx.y+threadIdx.y;
	if(indexX<size/2&&indexY<indexX){
		orient_square_GPU(dir,weights,thresholds,indexX*2,indexY*2,size);
	}
}

__global__ void multiply_kernel(bool *x, bool *dir,int size){//dfs
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ bool shared[];
	bool *xs=&shared[0];
	bool *ys=&shared[size];
	__shared__ bool flag[1];
	int j=index;
	// matrix multiplication variable
	while(j<size) {
		xs[j]=x[j];
		ys[j]=x[j];
		j+=1024;
	}
	flag[0]=true;
	__syncthreads();
	while(flag[0]){
		flag[0]=false;
		__syncthreads();
		j=index;
		while(j<size){
			if(xs[j]==1){
				j+=1024;
				continue;
			}
			for(int i=0;i<size;++i){
				if(dir[(i*size)+j]&xs[i]==1){
					ys[j]=1;
					flag[0]=true;
					break;
				}
			}
			j+=1024;
		}
		__syncthreads();
		j=index;
		while(j<size){
			xs[j]=ys[j];
			j+=1024;
		}
		__syncthreads();
	}
	j=index;
	while(j<size){
		x[j]=ys[j];
		j+=1024;
	}
}

/*
----------------------------non-worker-kernel---------------------------
*/

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

void Agent::up_GPU(vector<bool> signal){
	for(int i=0;i<measurableSize;++i) Gsignal[i]=signal[i];
	cudaMemcpy(dev_signal,Gsignal,measurableSize*sizeof(bool),cudaMemcpyHostToDevice);
	cudaMemset(dev_affected_worker,0,workerSize*sizeof(bool));
	multiply_kernel<<<1, 512, 2*measurableSize*sizeof(bool)>>>(dev_worker,dev_signal,dev_affected_worker,workerSize,measurableSize);

	cudaMemcpy(Gsignal,dev_signal,measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
	cudaMemcpy(Gaffected_worker,dev_affected_worker,workerSize*sizeof(bool),cudaMemcpyDeviceToHost);
}

//before invoke this function make sure dev_load and dev_signal have correct data
//the computed data will be in dev_load
void Agent::propagate_GPU(){//propagate
	logging_info->record_start();

	if(is_worker_solution){
		cudaMemset(dev_affected_worker,0,workerSize*sizeof(bool));
		multiply_kernel<<<1, 512, 2*measurableSize*sizeof(bool)>>>(dev_worker,dev_load,dev_affected_worker,workerSize,measurableSize);
		multiply_kernel<<<1, 512, 2*measurableSize*sizeof(bool)>>>(dev_worker,dev_signal,dev_affected_worker,workerSize,measurableSize);
		// standard operations
		disjunction_kernel<<<(measurableSize+255)/256,256>>>(dev_load,dev_signal,measurableSize);
		negate_disjunction_star_kernel<<<(measurableSize+255)/256,256>>>(dev_load,dev_signal,measurableSize);
	
		cudaMemcpy(Gload,dev_load,measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
		cudaMemcpy(Gaffected_worker,dev_affected_worker,workerSize*sizeof(bool),cudaMemcpyDeviceToHost);
	}
	else{
		multiply_kernel<<<1, 1024, 2*measurableSize*sizeof(bool)>>>(dev_load, dev_dir, measurableSize);
		multiply_kernel<<<1, 1024, 2*measurableSize*sizeof(bool)>>>(dev_signal, dev_dir, measurableSize);

		// standard operations
		disjunction_kernel<<<(measurableSize+255)/256,256>>>(dev_load,dev_signal,measurableSize);
		negate_disjunction_star_kernel<<<(measurableSize+255)/256,256>>>(dev_load,dev_signal,measurableSize);
		cudaMemcpy(Gload,dev_load,measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
	}

	logging_info->record_stop(logging::PROPAGATION);
}

void Agent::setSignal(vector<bool> observe){//this is where data comes in in every frame
	for(int i=0;i<observe.size();++i){
		Gobserve[i]=observe[i];
	}
	cudaMemcpy(dev_observe,Gobserve,measurableSize*sizeof(bool),cudaMemcpyHostToDevice);
}

void Agent::update_weights(){}

void Agent_Empirical::update_weights(){
	logging_info->record_start();

	if(is_worker_solution){
		update_weights_kernel_empirical<<<(workerSize+255)/256,256>>>(dev_worker,dev_observe,workerSize);
	}
	else{
		dim3 dimGrid2((measurableSize+15)/16,(measurableSize+15)/16);
		dim3 dimBlock2(16,16);
		update_weights_kernel<<<dimGrid2,dimBlock2>>>(dev_weights,dev_observe,measurableSize);
	}

	logging_info->record_stop(logging::UPDATE_WEIGHT);
}

void Agent_Distributed::update_weights(){
	logging_info->record_start();

	cudaMemset(dev_sensor_value,0.0,measurableSize*sizeof(float));
	calculate_sensor_value<<<(workerSize+255)/256,256>>>(dev_worker,dev_sensor_value,workerSize);
	update_weights_kernel_distributed<<<(workerSize+255)/256,256>>>(dev_worker,dev_sensor_value,dev_observe,workerSize,sensorSize,t);

	logging_info->record_stop(logging::UPDATE_WEIGHT);
}

void Agent_Discounted::update_weights(){
	logging_info->record_start();

	update_weights_kernel_discounted<<<(workerSize+255)/256,256>>>(dev_worker,dev_observe,workerSize);

	logging_info->record_stop(logging::UPDATE_WEIGHT);
}

void Agent::orient_all(){
	logging_info->record_start();

	if(is_worker_solution)
		orient_all_kernel<<<(workerSize+255)/256,256>>>(dev_worker,workerSize);
	else{
		dim3 dimGrid1((measurableSize/2+15)/16,(measurableSize/2+15)/16);
		dim3 dimBlock1(16,16);
		orient_all_kernel<<<dimGrid1,dimBlock1>>>(dev_dir,dev_weights,dev_thresholds,measurableSize);
	}

	logging_info->record_stop(logging::ORIENT_ALL);
}

void logging::record_start(){
	if(!using_log) return;
	cudaEventRecord(start);
}

void logging::record_stop(int LOG_TYPE){
	if(!using_log) return;

	float dt=0;
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&dt, start,stop);
	switch(LOG_TYPE){
	case logging::UPDATE_WEIGHT:
		n_update_weight++;
		t_update_weight+=dt;
		break;
	case logging::ORIENT_ALL:
		n_orient_all++;
		t_orient_all+=dt;
		break;
	case logging::PROPAGATION:
		n_propagation++;
		t_propagation+=dt;
		break;
	}
}

void Agent::update_state_GPU(bool mode){//true for decide	
	update_weights();
	t++;//when distributed or multiply agents, need to move it to upper loop
	//update_weight
	
	if(mode){
		orient_all();
	}//orient_all

	cudaMemcpy(dev_signal,dev_observe,measurableSize*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load,false,measurableSize*sizeof(bool));
	propagate_GPU();
	cudaMemcpy(Gcurrent,dev_load,measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_current,dev_load,measurableSize*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemcpy(Gdir,dev_dir,measurableSize*measurableSize*sizeof(bool),cudaMemcpyDeviceToHost);
}

void Agent::halucinate_GPU(vector<int> &actions_list){
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
	delete[] Gsignal;
	delete[] Gload;

	delete[] Gmask;
	delete[] Gcurrent;
	delete[] Gworker;
	delete[] Gaffected_worker;

	cudaFree(dev_dir);
	cudaFree(dev_thresholds);
	cudaFree(dev_weights);
	cudaFree(dev_observe);
	cudaFree(dev_signal);
	cudaFree(dev_load);

	cudaFree(dev_mask);
	cudaFree(dev_current);
	cudaFree(dev_worker);
	cudaFree(dev_affected_worker);

	cudaFree(dev_sensor_value);
}

void Agent::copy_sensors_name(vector<string> &sensors_names){
	this->sensors_names=sensors_names;
	for(int i=0;i<measurableSize;++i){
		name_to_num[sensors_names[i]]=i;
	}
	logging_info->append_log(logging::INIT,"Agent Sensors Name From Python, Size: "+to_string(sensors_names.size())+"\n");
}

void Agent::copy_context(vector<vector<int> > &context_key,vector<int> &context_value){
	for(int i=0;i<context_key.size();++i){
		context[pair<int,int>(context_key[i][0],context_key[i][1])]=context_value[i];
	}
	logging_info->append_log(logging::INIT,"Agent Context From Python, Context Pair Size: "+to_string(context_key.size())+"\n");
}

void Agent::copy_evals_name(vector<string> &evals_names){
	this->evals_names=evals_names;
	logging_info->append_log(logging::INIT,"Agent Evals Names From Python, Size "+to_string(evals_names.size())+"\n");
}

void Agent::copy_generalized_actions(vector<vector<int> > &generalized_actions){
	this->generalized_actions=generalized_actions;
	logging_info->append_log(logging::INIT,"Agent Generalized Action From Python, Size "+to_string(generalized_actions.size())+"\n");
}

void Agent::copy_size(int sensorSize){
	this->sensorSize=sensorSize;
	this->measurableSize=2*sensorSize;
	this->workerSize=sensorSize*(sensorSize-1)/2;
	logging_info->append_log(logging::INIT,"Agent Sensor Size From Python: "+to_string(sensorSize)+"\n");
	logging_info->append_log(logging::INIT,"Agent Measurable Size From Python: "+to_string(measurableSize)+"\n");
	logging_info->append_log(logging::INIT,"Agent Worker Size From Python: "+to_string(workerSize)+"\n");
}

void Agent::copy_direction(vector<vector<bool> >&data){
	Gdir=new bool[measurableSize*measurableSize];
	logging_info->append_log(logging::INIT,"Direction Matrix Generated, Size of Matrix: "
		+to_string(measurableSize)+"*"+to_string(measurableSize)+"\n");

	cudaMalloc(&dev_dir,measurableSize*measurableSize*sizeof(bool));
	logging_info->append_log(logging::INIT,"GPU Memory Malloced for Direction Matrix :"
		+to_string(measurableSize*measurableSize*sizeof(bool))+"Bits \n");

	for(int i=0;i<measurableSize;++i){
		for(int j=0;j<measurableSize;++j){
			if(data.empty()){
				Gdir[i*measurableSize+j]=false;
				if(i == j)
					Gdir[i*measurableSize+j] = true;
			}
			else{
				if(i<j) Gdir[ind(i,j,measurableSize)]=data[j][i];
				else Gdir[ind(i,j,measurableSize)]=data[i][j];
			}
		}
	}
	cudaMemcpy(dev_dir,Gdir,measurableSize*measurableSize*sizeof(bool),cudaMemcpyHostToDevice);
	logging_info->append_log(logging::INIT,"Direction Matrix Copied to GPU\n");
}

void Agent::copy_weight(vector<vector<double> >&data){
	Gweights=new double[measurableSize*measurableSize];
	logging_info->append_log(logging::INIT,"Weight Matrix Generated, Size of Matrix: "
		+to_string(measurableSize)+"*"+to_string(measurableSize)+"\n");

	cudaMalloc(&dev_weights,measurableSize*measurableSize*sizeof(double));
	logging_info->append_log(logging::INIT,"GPU Memory Malloced for Weight Matrix :"
		+to_string(measurableSize*measurableSize*sizeof(double))+"Bits \n");

	for(int i=0;i<measurableSize;++i){
		for(int j=0;j<measurableSize;++j){
			if(data.empty()){
				Gweights[ind(i,j,measurableSize)]=0.0;
			}
			else{
				if(i<j) Gweights[ind(i,j,measurableSize)]=data[j][i];
				else Gweights[ind(i,j,measurableSize)]=data[i][j];
			}
		}
	}
	cudaMemcpy(dev_weights,Gweights,measurableSize*measurableSize*sizeof(double),cudaMemcpyHostToDevice);
	logging_info->append_log(logging::INIT,"Weight Matrix Copied to GPU\n");
}
	
void Agent::copy_thresholds(vector<vector<double> >&data){
	Gthresholds=new double[measurableSize*measurableSize];
	logging_info->append_log(logging::INIT,"Thresholds Matrix Generated, Size of Matrix: "
		+to_string(measurableSize)+"*"+to_string(measurableSize)+"\n");

	cudaMalloc(&dev_thresholds,measurableSize*measurableSize*sizeof(double));
	logging_info->append_log(logging::INIT,"GPU Memory Malloced for Thresholds Matrix :"
		+to_string(measurableSize*measurableSize*sizeof(double))+"Bits \n");

	for(int i=0;i<measurableSize;++i){
		for(int j=0;j<measurableSize;++j){
			Gthresholds[i*measurableSize+j]=threshold;
		}
	}
	cudaMemcpy(dev_thresholds,Gthresholds,measurableSize*measurableSize*sizeof(double),cudaMemcpyHostToDevice);
	logging_info->append_log(logging::INIT,"Thresholds Matrix Copied to GPU\n");
}

void Agent::copy_name(string name){
	this->name=name;
	logging_info->append_log(logging::INIT,"Agent Name From Python: "+name+"\n");
}

void Agent::reset_time(int n){
	this->t=n;
	logging_info->append_log(logging::INIT,"Time Reset to :"+to_string(n)+"\n");

	srand (time(NULL));
	logging_info->append_log(logging::INIT,"Time Seed Set \n");
}

void logging::createEvent(){
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

void Agent::gen_worker(){
	Gworker=new worker[workerSize];
	logging_info->append_log(logging::INIT,"Worker Generated, Number of Worker: "+to_string(workerSize)+"\n");

	initWorkerMemory(dev_weights,dev_dir);
	cudaMalloc(&dev_worker,workerSize*sizeof(worker));
	logging_info->append_log(logging::INIT,"GPU Memory Malloced for Worker :"+to_string(workerSize*sizeof(worker))+"Bits \n");

	cudaMemcpy(dev_worker,Gworker,workerSize*sizeof(worker),cudaMemcpyHostToDevice);
	logging_info->append_log(logging::INIT,"Worker Copied to GPU\n");
}

void Agent::gen_other_parameters(){
	Gobserve=new bool[measurableSize];
	Gsignal=new bool[measurableSize];
	Gload=new bool[measurableSize];
	Gmask=new bool[measurableSize];
	Gcurrent=new bool[measurableSize];
	Gaffected_worker=new bool[workerSize];
	logging_info->append_log(logging::INIT,"Other Parameter Generated\n");
	
	cudaMalloc(&dev_observe,measurableSize*sizeof(bool));
	cudaMalloc(&dev_signal,measurableSize*sizeof(bool));
	cudaMalloc(&dev_load,measurableSize*sizeof(bool));
	cudaMalloc(&dev_mask,measurableSize*sizeof(bool));
	cudaMalloc(&dev_current,measurableSize*sizeof(bool));
	cudaMalloc(&dev_affected_worker,workerSize*sizeof(bool));
	cudaMalloc(&dev_sensor_value,measurableSize*sizeof(float));
	logging_info->append_log(logging::INIT,"GPU Memory Malloced for Other Parameter :"+
		to_string(5*measurableSize*sizeof(bool)+workerSize*sizeof(bool)+measurableSize*sizeof(float))+"\n");
}

void Agent::initData(string name,int sensorSize,vector<vector<int> > context_key,vector<int> context_value,
		vector<string> sensors_names,vector<string> evals_names,vector<vector<int> > generalized_actions){
	//data init
	logging_info->append_log(logging::INIT,"----------------Data Initialization----------------\n");

	reset_time(0);
	vector<vector<double> > tmp_weight,tmp_thresholds;
	vector<vector<bool> > tmp_direction;
	if(Gdir!=NULL) freeData();

	copy_name(name);
	copy_size(sensorSize);
	copy_sensors_name(sensors_names);
	copy_evals_name(evals_names);
	copy_generalized_actions(generalized_actions);
	copy_context(context_key,context_value);
	
	copy_weight(tmp_weight);
	copy_direction(tmp_direction);
	copy_thresholds(tmp_thresholds);
	if(is_worker_solution) gen_worker();
	gen_other_parameters();

	logging_info->append_log(logging::INIT,"----------------Data Initialization----------------\n\n");
}

void Agent::initNewSensor(vector<vector<int> >&list){
	vector<vector<double> > tmp_weight=addSensors(list);
	vector<vector<double> > tmp_thresholds;
	vector<vector<bool> > tmp_direction;
	if(Gdir!=NULL) freeData();

	copy_name(name);
	copy_size(sensorSize);
	copy_sensors_name(sensors_names);
	
	copy_weight(tmp_weight);
	copy_direction(tmp_direction);
	copy_thresholds(tmp_thresholds);
	if(is_worker_solution) gen_worker();
	gen_other_parameters();
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

vector<vector<double> > Agent::getVectorWeight(){
	vector<vector<double> > result;
	for(int i=0;i<measurableSize;++i){
		vector<double> tmp;
		result.push_back(tmp);
	}
	cudaMemcpy(Gweights,dev_weights,measurableSize*measurableSize*sizeof(double),cudaMemcpyDeviceToHost);
	int x=1,y=0;
	while(y<measurableSize-1){
		result[x].push_back(Gweights[ind(y,x,measurableSize)]);
		x++;
		if(x>=measurableSize){
			y++;
			x=y+1;
		}
	}
	int index=measurableSize/2,index_c=index+1;
	if(index%2==1){
		index--;
		index_c--;
	}
	for(int i=0;i<measurableSize;++i){
		double value=0.0;
		if(index==i||index_c==i){
			value+=result[i][index-2];
			value+=result[i][index_c-2];
		}
		else{
			if(index>i) value+=result[index][i];
			else if(index<i) value+=result[i][index];
			if(index_c>i) value+=result[index_c][i];
			else if(index_c<i) value+=result[i][index_c];
		}
		result[i].push_back(value);
	}
	return result;
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