using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.IO
{
    interface LayerWriter
    {
        void WriteLayer(Layer l);
        void WriteLayer(ConvolutionLayer l);
        void WriteLayer(FullyConnectedLayer l);
        void WriteLayer(InputLayer l);
        void WriteLayer(PoolingLayer l);
        void WriteLayer(ReLuLayer l);
        void WriteLayer(SoftMaxLayer l);
        void WriteNumber(int n);
    }
}
