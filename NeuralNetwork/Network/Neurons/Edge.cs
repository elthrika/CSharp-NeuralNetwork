namespace NeuralNetwork
{
    internal class Edge
    {
        public double Weight { get; set; }

        public Neuron Destination { get; private set; }
        public Neuron Origin { get; private set; }

        public Edge(double weight, Neuron origin, Neuron dest)
        {
            Weight = weight;
            Destination = dest;
            Origin = origin;
            Origin.AddOutEdge(this);
            Destination.AddInEdge(this);
        }

        internal void UpdateWeight(double targetvalue)
        {
            double delta = Destination.GetDelta(targetvalue, Destination);
            double diffw = -Network.LEARNING_RATE * delta * Origin.Value;

            Weight += diffw;
        }

        public override string ToString()
        {
            return $"{Origin.ID} --{Weight}-> {Destination.ID}";
        }
    }
}
