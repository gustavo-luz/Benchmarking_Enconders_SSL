| **Technique** 	| **Backbone**           	| **Prediction Head**              	| **Freeze** 	| **Stage** 	| **File**                        	|
|---------------	|------------------------	|----------------------------------	|------------	|-----------	|---------------------------------	|
| CPC           	| Convolutional1DEncoder 	| MLP (7600, 128, 6)               	| False      	| finetune  	| finetune/cpc.yaml               	|
| CPC           	| Convolutional1DEncoder 	| MLP (7600, 128, 6)               	| True       	| finetune  	| finetune/cpc_freeze.yaml        	|
| CPC           	| ResNetEncoder          	| MLP (1792, 128, 6)               	| False      	| finetune  	| finetune/cpc_resnet.yaml        	|
| CPC           	| ResNetEncoder          	| MLP (1792, 128, 6)               	| True       	| finetune  	| finetune/cpc_resnet_freeze.yaml 	|
| TFC           	| TFC_Backbone           	| TFC_PredicionHead (128*2, 64, 6) 	| False      	| finetune  	| finetune/tfc.yaml               	|
| TFC           	| TFC_Backbone           	| TFC_PredicionHead (128*2, 64, 6) 	| True       	| finetune  	| finetune/tfc_freeze.yaml        	|
| TFC           	| _ResNet1D              	| MLP (256, 128, 6)                	| False      	| finetune  	| finetune/tfc_resnet.yaml        	|
| TFC           	| _ResNet1D              	| MLP (256, 128, 6)                	| True       	| finetune  	| finetune/tfc_resnet_freeze.yaml 	|
| TNC           	| TSEncoder              	| MLP (320, 128, 6)                	| False      	| finetune  	| finetune/tnc_ts2vec.yaml        	|
| TNC           	| TSEncoder              	| MLP (320, 128, 6)                	| True       	| finetune  	| finetune/tnc_ts2vec_freeze.yaml 	|
| TNC           	| _ResNet1D              	| MLP (28, 128, 6)                 	| False      	| finetune  	| finetune/tnc_resnet.yaml        	|
| TNC           	| _ResNet1D              	| MLP (28, 128, 6)                 	| True       	| finetune  	| finetune/tnc_resnet_freeze.yaml 	|
| SUPERVISED    	| ResNet1D_8             	| -                                	| -          	| train     	| train/resnet1d8_6classes.yaml   	|
| SUPERVISED    	| ResNet1D_5             	| -                                	| -          	| train     	| train/resnet1d5_6classes.yaml   	|
| SUPERVISED    	| ResNetSE1D_8           	| -                                	| -          	| train     	| train/resnetse1d8_6classes.yaml 	|
| SUPERVISED    	| ResNetSE1D_5           	| -                                	| -          	| train     	| train/resnetse1d5_6classes.yaml 	|