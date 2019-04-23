using System.IO;

namespace NeuralNetwork
{
    internal class InputLayer : Layer
    {
        internal double[] inputValues;

        public InputLayer(int size, Network n) : base(LayerType.InputLayer, n)
        {
            Size = size;
            AllNeurons = new Neuron[Size];
            for (int i = 0; i < Size; i++)
            {
                AllNeurons[i] = Source.GetInputNeuron(this);
            }
        }

        public void AddInput(double[] inputs)
        {
            inputValues = inputs;
        }

        public double[] GetInput()
        {
            return inputValues;
        }

        internal override void EvaluateAllNeurons()
        {
            int i = 0;
            foreach (var neuron in AllNeurons)
            {
                neuron.SetValue(inputValues[i++]);
            }
        }

        internal override void Backpropagate(double[] targetvalues = null)
        {
            // Do nothing on this layer
        }

        public override string ToString()
        {
            return $"Input Layer of Size: {Size}";
        }
    }
}
