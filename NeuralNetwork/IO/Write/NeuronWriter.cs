using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.IO
{
    interface NeuronWriter
    {
        void WriteNeuron(Neuron n);
        void WriteNeuron(ActivationNeuron n);
        void WriteNeuron(BiasNeuron n);
        void WriteNeuron(ConvolutionNeuron n);
        void WriteNeuron(InputNeuron n);
        void WriteNeuron(PoolingNeuron n);
        void WriteNeuron(SoftMaxNeuron n);
    }
}
