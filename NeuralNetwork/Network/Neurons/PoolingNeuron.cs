using System;
using System.IO;

namespace NeuralNetwork
{
    internal class PoolingNeuron : Neuron
    {
        internal int MaxIdx { get; set; }

        internal PoolingNeuron(Layer parent, long id) : base(NeuronType.PoolingNeuron, parent, id)
        {

        }

        internal override double GetDelta(double targetvalue, Neuron caller = null)
        {
            if (caller == null) throw new ArgumentNullException();
            if (caller == this) throw new ArgumentException();
            base.GetDelta(targetvalue, caller);
            if (inEdges[MaxIdx].Origin == caller) return _delta;
            return 0.0d;
        }

        internal override void Backpropagate(double targetvalue)
        {
            _delta = 0;
            if (outEdges.Count == 0)
            {
                _delta = Parent.Parent.dNetworkLossFunction(targetvalue, Value);
            }
            else
            {
                foreach (var edge in outEdges)
                {
                    _delta += edge.Weight * edge.Destination.GetDelta(targetvalue, this);
                }
            }
        }
        
        internal override void WriteToFilePlainText(StreamWriter sw)
        {
            sw.WriteLine(Type.ToString());
            sw.WriteLine(ID);
        }
    }
}
