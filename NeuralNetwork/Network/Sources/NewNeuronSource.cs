using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    class NewNeuronSource : NeuronSource
    {
        long curID = 0;

        Dictionary<long, Neuron> AllGenerated = new Dictionary<long, Neuron>();
        List<Edge> Edges = new List<Edge>();
        
        public IReadOnlyDictionary<long, Neuron> GetGeneratedNeurons()
        {
            return AllGenerated;
        }

        public IReadOnlyList<Edge> GetGeneratedEdges()
        {
            return Edges.AsReadOnly();
        }

        public ActivationNeuron GetActivationNeuron(ActivationFunctionType funtype, bool scale = true, long id = -1)
        {
            var n = new ActivationNeuron(funtype, scale, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public BiasNeuron GetBiasNeuron(double constantVal = 1, long id = -1)
        {
            var n = new BiasNeuron(constantVal, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public ConvolutionNeuron GetConvolutionNeuron(ConvolutionLayer myLayer, long id = -1)
        {
            var n = new ConvolutionNeuron(myLayer, GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public InputNeuron GetInputNeuron(long id = -1)
        {
            var n = new InputNeuron(GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public PoolingNeuron GetPoolingNeuron(long id = -1)
        {
            var n = new PoolingNeuron(GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public SoftMaxNeuron GetSoftMaxNeuron(long id = -1)
        {
            var n = new SoftMaxNeuron(GetID(id));
            AllGenerated.Add(n.ID, n);
            return n;
        }

        public long GetID(long id)
        {
            curID++;
            return id > 0 ? id : curID;
        }

        public Edge MakeEdge(double weight, Neuron origin, Neuron destination)
        {
            Edge e = new Edge(weight, origin, destination);
            Edges.Add(e);
            return e;
        }

        public Edge MakeEdge(double weight, long originID, long destinationID)
        {
            var origin = AllGenerated[originID];
            var destination = AllGenerated[destinationID];
            return MakeEdge(weight, origin, destination);
        }
    }
}
