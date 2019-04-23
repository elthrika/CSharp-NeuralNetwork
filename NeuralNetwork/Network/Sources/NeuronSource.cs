using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal interface NeuronSource
    {
        ActivationNeuron GetActivationNeuron(Layer parent, ActivationFunctionType funtype, bool scale = true, long id = -1);
        BiasNeuron GetBiasNeuron(Layer parent, double constantVal = 1.0d, long id = -1);
        ConvolutionNeuron GetConvolutionNeuron(Layer parent, long id = -1);
        InputNeuron GetInputNeuron(Layer parent, long id = -1);
        PoolingNeuron GetPoolingNeuron(Layer parent, long id = -1);
        SoftMaxNeuron GetSoftMaxNeuron(Layer parent, long id = -1);

        Edge MakeEdge(double weight, Neuron origin, Neuron destination);
        Edge MakeEdge(double weight, long originID, long destinationID);

        IReadOnlyDictionary<long, Neuron> GetGeneratedNeurons();
        IReadOnlyList<Edge> GetGeneratedEdges();

        long GetID(long id);
    }
}
