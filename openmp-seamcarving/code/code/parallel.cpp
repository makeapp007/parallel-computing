#include <iostream>
#include <fstream>
#include <math.h> 
#include <time.h>

#include <omp.h>

using namespace std;
// Readme
// I didnot use malloc since it is time consuming to use malloc to allocate and free while storing data on the stack is fast and enough.
// if the image is bigger than 1000*1000, for example, 2000*2000
// please set im, tmp, seam size to 2000*2000 correspondingly.
// Usage:
// input: data2.txt, storing pixel value.
// output: data_seam.txt, storing pixel which has been performed seam carving.

// record the source image
int im[7000][10000];
// record the energy
int tmp[7000][10000];
// recording the seam
int seam[7000];


FILE* stream;


int main()
{
	stream = fopen("9k.txt","r");
	int h,w;
	int element;


	if(stream==0){
		cout<<"ERROR, file does not exist"<<endl;
		return 0;
	}

	fscanf(stream,"%d",&h);
	fscanf(stream,"%d",&w);


	cout<<w<<" "<<h<<endl;
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			fscanf(stream,"%d",&element);
			im[i][j]=element;
		}
	}
	// for(int i=0;i<h;i++){
	// 	for(int j=0;j<w;j++){
	// 		cout<<im[i][j]<<" ";
	// 	}
	// 	cout<<endl;
	// }

	ofstream outfile;
	outfile.open("data_seam.txt");
    double wtime = omp_get_wtime ( );

	int totround=w/2;
	for(int round=0;round<totround;round++){

		#pragma omp parallel for 
		for (int i = 0; i < h; ++i)
		{
			/* code */
			tmp[i][w-1]=100000;			
			tmp[i][0]=100000;
		}
		#pragma omp parallel for 
		for (int j = 0; j < w; ++j)
		{
			/* code */
			tmp[0][j]=100000;			
			tmp[h-1][j]=100000;
		}

		// calculating gradient
		#pragma omp parallel for 
		for(int i=1;i<h-1;i++){
			#pragma omp parallel for 
			for(int j=1;j<w-1;j++){
				// set the margin to a long number, at least long than 255
				tmp[i][j]=int(sqrt((im[i][j+1]-im[i][j-1])*(im[i][j+1]-im[i][j-1])+(im[i+1][j]-im[i-1][j])*(im[i+1][j]-im[i-1][j])));
			}
		}

		// calculate energy
		// int tmp1,tmp2,tmp3;
		for(int i=2;i<h-1;i++){
			#pragma omp parallel for 
			for(int j=1;j<w-1;j++){

				int tmp1=tmp[i-1][j-1];
				int tmp2=tmp[i-1][j];
				int tmp3=tmp[i-1][j+1];
				tmp[i][j]=min(min(tmp1,tmp2),tmp3)+tmp[i][j];
			}
		}
		// cout<<endl;
		// 	for(int i=0;i<h;i++){
		// 		for(int j=0;j<w;j++){
		// 			cout<<tmp[i][j]<<" ";
		// 		}
		// 		cout<<endl;
		// 	}
		// cout<<tmp[h-2][w-2]<<endl;

		// find min in last row
		int min=10000,min_index=0;	
		for(int j=0;j<w;j++){
			if(min>tmp[h-2][j]){
				min=tmp[h-2][j];
				min_index=j;
			}
		}
		seam[h-2]=min_index;
		// cout<<"min_index  "<<min_index<<endl;
		// trace back to find the seam
		for(int i=h-3;i>1;i--){
			int tmp1=tmp[i][min_index-1];
			int tmp2=tmp[i][min_index];
			int tmp3=tmp[i][min_index+1];
			if(tmp3>tmp1 && tmp2>tmp1) seam[i]=min_index-1;
			else if(tmp1>=tmp2&& tmp3>=tmp2) seam[i]=min_index;
			else if(tmp2>=tmp3&& tmp1>=tmp3) seam[i]=min_index+1;
			min_index=seam[i];
		}

		// for(int i=0;i<h;i++){
		// 	cout<<seam[i]<<" ";
		// }
		// cut the seam in the image
		#pragma omp parallel for 
		for(int i=0;i<h;i++){
			// int start=seam[i];
			for(int j=seam[i];j<w;j++){
				im[i][j]=im[i][j+1];
			}
		}
		// cout<<endl;
		// for(int i=0;i<h;i++){
		// 	for(int j=0;j<w;j++){
		// 		// fscanf(stream,"%d",&w);
		// 		cout<<im[i][j]<<" ";
		// 	}
		// 	cout<<endl;
		// }


		w--;
		// cout<<w<<endl;
	}

    wtime = omp_get_wtime ( ) - wtime;
    printf ( "  Parallel %15.10f\n ms\n",wtime );



	// write im to file
	for(int i=0;i<h;i++){
		for(int j=0;j<w;j++){
			outfile << im[i][j]<< " ";
		}
		outfile<<endl;
	}

	outfile.close();
	return 0;
}




