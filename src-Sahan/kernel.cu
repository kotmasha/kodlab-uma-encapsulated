#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "Snapshot.h"
#include "Agent.h"

//CUDA introduction
/*
in CUDA:
__host__ means the function happens on CPU
__device__ means the function happens on Device(GPU)
__host__ __device__ means the function can be used both on CPU and GPU(the compi on both CPU and GPU)
__global__ means it is an global function. global function is where parallel happens.
	Many threads runs the same code in global function, the way to distinguish different thread is by block and thread
	blockIdx means block ID, threadIdx means thread ID. In CUDA, many threads form a block, many blocks form a grid, you can find the detail in the ppt I sent you
	blockDim is the dimension of block. block and thread can be in multiply dimension(see in ppt)
	to call a global function, you need to specify block number and thread num within each block like:
		fun<<<blockNum,threadNum>>>(para1,para2...), blockNum and threadNum can be in multiply dimension
	when you are accessing the data, make sure they do not go beyond boundary(that is why I have lots of "size" variable to check in almost every global function)
	in global function you can use __device__ function or __host__ __device__ function as long as EVERY VARIABLE IS ON GPU. This is a very strict rule, you cannot access CPU memory on GPU, neither can you access GPU memory on CPU.
cudaMalloc(&variable_address,size*sizeof(data_type)):
	The function is used to malloc space for variable on GPU
cudaMemcpy(&copy_to_address,&copy_from_address,size*sizeof(data_type),tag):
	The function is copy data from one place to another.
	tag has: cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice,cudaMemcpyHostToHost
cudaMemset(&variable_address,value,size*sizeof(data_type)):
	The function is like memset in C++, give the same value to an address of data
cudaFree(&variable_address):
	The function is used to free a variable, like delete in C++

I believe those functions above is enough for the project, if you have any question just email me
*/

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

__global__ void stream_compaction_kernel(int *scan,bool *observe,int *result,int size){
	int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index<size&&observe[index]){
		result[scan[index]]=index;
	}
}

//helper function

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
	//extern __shared__ int shared[];
	//shared[indexX]=-1;
	//shared[indexY]=-1;
	//__syncthreads();
	if(indexX<size&&indexY<size){
		/*if(shared[indexX]==-1){
			shared[indexX]=observe[indexX];
		}
		if(shared[indexY]==-1){
			shared[indexY]=observe[indexY];
		}
		int x=shared[indexX];
		int y=shared[indexY];*/
		weights[ind(indexY,indexX,size)]+=observe[indexX]*observe[indexY];
		//weights[ind(indexY,indexX,size)]+=x*y;
	}
}

__global__ void update_weights_kernels(double *weights,int *result,int s,int size){
	int indexX=blockDim.x*blockIdx.x+threadIdx.x;
	int indexY=blockDim.y*blockIdx.y+threadIdx.y;
	if(indexX<s&&indexY<s){
		int x=result[indexX];
		int y=result[indexY];
		weights[ind(y,x,size)]+=1;
	}
}

__global__ void orient_all_kernel(bool *dir,double *weights,double *thresholds,int size){
	int indexX=blockDim.x*blockIdx.x+threadIdx.x;
	int indexY=blockDim.y*blockIdx.y+threadIdx.y;
	//the commented code is the optimization for the triangle problem we discussed. I think the speed is fast for now so I just use the original one
	/*if(indexX<size){//possible optimazation
		if(indexY>indexX) orient_square_GPU(dir,weights,thresholds,2*(size/2-1-indexX),2*(size/2-1-indexY),size);
		else if(indexY<indexX) orient_square_GPU(dir,weights,thresholds,2*indexX,2*indexY,size);
	}*/
	if(indexX<size/2&&indexY<indexX){
		orient_square_GPU(dir,weights,thresholds,indexX*2,indexY*2,size);
	}
}

// this is a standard implementation of matrix and vector multiplication 
__global__ void multiply_kernel(int *x, int *dir, int *y,int size){//dfs
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	extern __shared__ int shared[];
	int *xs=&shared[0];
	int *ys=&shared[size];
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
		y[j]=ys[j];
		j+=1024;
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
void Snapshot::propagate_GPU(){//propagate
	bool2int_kernel<<<(size+255)/256,256>>>(tmp_load,dev_load,size);
	bool2int_kernel<<<(size+255)/256,256>>>(tmp_signal,dev_signal,size);

	// not copy the matrix
	// find a different way
	// THIS IS FOR LATER USE
	bool2int_kernel<<<(size*size+255)/256,256>>>(tmp_dir, dev_dir, size*size);
	//cout<<size<<endl;
	// set out_load vector to 0
	//cudaMemset(out_load, 0, size*sizeof(int));
	multiply_kernel<<<1, 1024, 2*size*sizeof(int)>>>(tmp_load, tmp_dir, out_load, size);
	cudaMemcpy(tmp_load, out_load, size*sizeof(int), cudaMemcpyDeviceToDevice);

	// out_load must have the result for load
	int2bool_kernel<<<(size+255)/256,256>>>(dev_load, tmp_load, size);

	//cudaMemset(out_signal, 0, size*sizeof(int));
	multiply_kernel<<<1, 1024, 2*size*sizeof(int)>>>(tmp_signal, tmp_dir, out_signal, size);
	cudaMemcpy(tmp_signal, out_signal, size*sizeof(int), cudaMemcpyDeviceToDevice);

	// out_signal must have the result for signal
	int2bool_kernel<<<(size+255)/256, 256>>>(dev_signal, tmp_signal, size);

	// standard operations
	disjunction_kernel<<<(size+255)/256,256>>>(dev_load,dev_signal,size);
	negate_disjunction_star_kernel<<<(size+255)/256,256>>>(dev_load,dev_signal,size);
	
	cudaMemcpy(Gload,dev_load,size*sizeof(bool),cudaMemcpyDeviceToHost);
}

void Snapshot::setSignal(vector<bool> observe){//this is where data comes in in every frame
	for(int i=0;i<observe.size();++i){
		Gobserve[i]=observe[i];
	}
	cudaMemcpy(dev_observe,Gobserve,size*sizeof(bool),cudaMemcpyHostToDevice);
}

void Snapshot::update_state_GPU(bool mode){//true for decide
	bool2int_kernel<<<(size+255)/256,256>>>(dev_scan,dev_observe,size);
	thrust::exclusive_scan(thrust::device,dev_scan, dev_scan+size, dev_scan); // in-place scan
	int s=0;
	bool ss=0;
	
	cudaMemcpy(&s,dev_scan+size-1,sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(&ss,dev_observe+size-1,sizeof(bool),cudaMemcpyDeviceToHost);
	s+=ss;

	stream_compaction_kernel<<<(size+255)/256,256>>>(dev_scan,dev_observe,tmp_weight,size);

	dim3 dimGrid((s+15)/16,(s+15)/16);
	dim3 dimBlock(16,16);
	//cout<<s<<","<<size<<endl;
	update_weights_kernels<<<dimGrid,dimBlock>>>(dev_weights,tmp_weight,s,size);
	//cout<<s<<","<<size<<endl;
	/*dim3 dimGrid2((size+15)/16,(size+15)/16);
	dim3 dimBlock2(16,16);
	update_weights_kernel<<<dimGrid2,dimBlock2,size*sizeof(int)>>>(dev_weights,dev_observe,size);*/
	//update_weight
	
	if(mode){
		dim3 dimGrid1((size/2+15)/16,(size/2+15)/16);
		dim3 dimBlock1(16,16);
		orient_all_kernel<<<dimGrid1,dimBlock1>>>(dev_dir,dev_weights,dev_thresholds,size);
	}//orient_all

	cudaMemcpy(dev_signal,dev_observe,size*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemset(dev_load,false,size*sizeof(bool));
	propagate_GPU();
	cudaMemcpy(Gcurrent,dev_load,size*sizeof(bool),cudaMemcpyDeviceToHost);
	cudaMemcpy(dev_current,dev_load,size*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemcpy(Gdir,dev_dir,size*size*sizeof(bool),cudaMemcpyDeviceToHost);
}

void Snapshot::halucinate_GPU(vector<int> actions_list){
	//mask=Signal([(ind in actions_list) for ind in xrange(self._SIZE)])
	vector<bool> mask=initMask(actions_list);
	vector<int> v;
	for(int i=0;i<actions_list.size();++i){
		for(int j=0;j<size;++j){
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
	cudaMemcpy(dev_mask,Gmask,size*sizeof(bool),cudaMemcpyHostToDevice);
	//copy data
	cudaMemcpy(dev_signal,dev_mask,size*sizeof(bool),cudaMemcpyDeviceToDevice);
	cudaMemcpy(dev_load,dev_current,size*sizeof(bool),cudaMemcpyDeviceToDevice);
	propagate_GPU();
	//return self.propagate(mask,self._CURRENT)
}

void Snapshot::freeData(){//free data in case of memory leak
	delete[] Gdir;
	delete[] Gweights;
	delete[] Gthresholds;
	delete[] Gobserve;
	delete[] Gdfs;
	delete[] Gsignal;
	delete[] Gload;

	delete[] Gmask;
	delete[] Gcurrent;

	cudaFree(dev_dir);
	cudaFree(dev_thresholds);
	cudaFree(dev_weights);
	cudaFree(dev_observe);
	cudaFree(dev_dfs);
	cudaFree(dev_signal);
	cudaFree(dev_load);
	cudaFree(tmp_signal);
	cudaFree(tmp_load);
	
	// cudaFree(tmp_dir)
	cudaFree(tmp_dir);
	cudaFree(out_load);
	cudaFree(out_signal);

	cudaFree(dev_scan);
	cudaFree(tmp_weight);

	cudaFree(dev_mask);
	cudaFree(dev_current);
}

void Snapshot::initData(string name,int size,double threshold,vector<vector<int> > context_key,vector<int> context_value,
		vector<string> sensors_names,vector<string> evals_names,vector<vector<int> > generalized_actions){
	//data init
	this->name=name;
	this->size=size;
	this->threshold=threshold;
	this->sensors_names=sensors_names;
	this->evals_names=evals_names;
	this->generalized_actions=generalized_actions;
	srand (time(NULL));
	for(int i=0;i<size;++i){
		name_to_num[sensors_names[i]]=i;
	}
	if(Gdir!=NULL){
		freeData();
	}
	
	Gdir=new bool[size*size];
	Gweights=new double[size*size];
	Gthresholds=new double[size*size];
	Gobserve=new bool[size];
	Gdfs=new bool[1];
	Gsignal=new bool[size];
	Gload=new bool[size];

	Gmask=new bool[size];
	Gcurrent=new bool[size];
	
	cudaMalloc(&dev_dir,size*size*sizeof(bool));
	cudaMalloc(&dev_thresholds,size*size*sizeof(double));
	cudaMalloc(&dev_weights,size*size*sizeof(double));
	cudaMalloc(&dev_observe,size*sizeof(bool));
	cudaMalloc(&dev_dfs,sizeof(bool));
	cudaMalloc(&dev_signal,size*sizeof(bool));
	cudaMalloc(&dev_load,size*sizeof(bool));

	// cudaMalloc tmp_dir
	cudaMalloc(&tmp_dir, size*size*sizeof(int));
	cudaMalloc(&tmp_signal,size*sizeof(int));
	cudaMalloc(&tmp_load,size*sizeof(int));
	cudaMalloc(&out_signal, size*sizeof(int));
	cudaMalloc(&out_load, size*sizeof(int));

	cudaMalloc(&dev_mask,size*sizeof(bool));
	cudaMalloc(&dev_current,size*sizeof(bool));

	cudaMalloc(&dev_scan,size*sizeof(int));
	cudaMalloc(&tmp_weight,size*sizeof(int));

	// we need to make diagonals of Gdir = 1, true
	for(int i=0;i<size;++i){
		for(int j=0;j<size;++j){
			Gthresholds[i*size+j]=threshold;
			Gweights[i*size+j]=0.0;
			Gdir[i*size+j]=false;
			
			// new implementation
			if(i == j)
				Gdir[i*size+j] = true;
		}
	}

	cudaMemcpy(dev_thresholds,Gthresholds,size*size*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_weights,Gweights,size*size*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dir,Gdir,size*size*sizeof(bool),cudaMemcpyHostToDevice);
	//init threshold

	for(int i=0;i<context_key.size();++i){
		context[pair<int,int>(context_key[i][0],context_key[i][1])]=context_value[i];
	}
	cout<<"succeed"<<endl;
}

//those three functions down there are get functions for the variable in C++
vector<bool> Snapshot::getCurrent(){
	vector<bool> result;
	for(int i=0;i<size;++i){
		result.push_back(Gcurrent[i]);
	}
	return result;
}

vector<bool> Snapshot::getLoad(){
	vector<bool> result;
	for(int i=0;i<size;++i){
		result.push_back(Gload[i]);
	}
	return result;
}

vector<vector<bool> > Snapshot::getDir(){
	vector<vector<bool> > result;
	for(int i=0;i<size;++i){
		vector<bool> tmp;
		for(int j=0;j<size;++j){
			tmp.push_back(Gdir[i*size+j]);
		}
		result.push_back(tmp);
	}
	return result;
}