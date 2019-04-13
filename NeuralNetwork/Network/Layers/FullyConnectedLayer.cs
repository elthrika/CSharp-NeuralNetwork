using System.IO;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class FullyConnectedLayer : Layer
    {

        public bool Scaling { get; private set; }

        public FullyConnectedLayer(int size, Layer previousLayer, NeuronSource s, ActivationFunctionType functionType = ActivationFunctionType.Tanh, bool scale = true) 
            : base(LayerType.FullyConnected, s)
        {
            Scaling = scale;

            var prevNeurons = previousLayer.GetNeurons();
            Size = size;
            AllNeurons = new Neuron[Size+1]; // 1 bias neuron

            for (int i = 0; i < size; i++)
            {
                AllNeurons[i] = Source.GetActivationNeuron(functionType, Scaling);
            }
            BiasNeuron bias = Source.GetBiasNeuron();
            for (int i = 0; i < Size; i++)
            {
                Edge e = Source.MakeEdge(Helper.Random, bias, AllNeurons[i]);
            }
            AllNeurons[Size] = bias;

            foreach (var prevneuron in prevNeurons)
            {
                foreach (var myneuron in AllNeurons)
                {
                    Edge e = Source.MakeEdge(Helper.Random, prevneuron, myneuron);
                }
            }
        }
        
        public ActivationFunctionType GetActivationFunctionType()
        {
            return ((ActivationNeuron)AllNeurons[0]).ActivationFunType;
        }

        internal override void EvaluateAllNeurons()
        {
#if DEBUG
            foreach (var neuron in AllNeurons)
            {
                neuron.CalculateValue();
            }
#else
            Parallel.ForEach(AllNeurons, n => n.CalculateValue());
#endif
        }

        internal override void Backpropagate(double[] targetvalues = null)
        {
#if DEBUG
            for (int i = 0; i < Size; i++)
            {
                double targetvalue = targetvalues == null ? 0 : targetvalues[i];
                AllNeurons[i].Backpropagate(targetvalue: targetvalue);

            }
#else
            Parallel.For(0, Size, i =>
            {
                double targetvalue = targetvalues == null ? 0 : targetvalues[i];
                AllNeurons[i].Backpropagate(targetvalue: targetvalue);
            });
#endif
        }


        #region IO

        public override string ToString()
        {
            string s = $"Fully-Connected Layer of Size: {Size}";
            return s;
        }

        #endregion
    }
}
