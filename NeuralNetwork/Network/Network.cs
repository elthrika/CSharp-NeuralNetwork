using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
#if DEBUG
using System.Diagnostics;
#endif


namespace NeuralNetwork
{
    public class Network
    {
        public delegate double LossFunction(double a, double b);

        public enum LossFunctionType
        {
            ErrorSquared,
            Logistical,
            Hinge,
            SquareLossClassification,
            LogisticLoss,
            ExponentialLoss,
            CrossEntropyLoss,
            //CosineSimilarity
        }
        internal static LossFunction NetworkLossFunction;
        internal static LossFunction dNetworkLossFunction;
        internal LossFunctionType lossFunctionType { get; private set; }

        internal static Network Instance;

        internal NeuronSource Source { get; private set; }

        Stack<Layer> Layers;
        InputLayer InputLayer;
        Layer Outputlayer
        {
            get { return Layers.ElementAt(0); }
        }

        internal IEnumerable<Layer> GetLayers()
        {
            return Layers;
        }

        internal static double LEARNING_RATE = 1;

        /*
        ***************************** 
        ****CONSTRUCTOR**************
        *****************************
        */
        public Network(LossFunctionType lft = LossFunctionType.ErrorSquared)
        {
            Layers = new Stack<Layer>();
            Instance = this;

            Source = new NewNeuronSource();
            lossFunctionType = lft;

            switch (lft)
            {
                case LossFunctionType.ErrorSquared:
                    NetworkLossFunction = (target, result) => 0.5 * Math.Pow(target - result, 2);
                    dNetworkLossFunction = (target, result) => result - target;
                    break;
                case LossFunctionType.Logistical:
                    NetworkLossFunction = (target, result) => -target * Math.Log(result);
                    dNetworkLossFunction = (target, result) => -target / result;
                    break;
                case LossFunctionType.Hinge:
                    NetworkLossFunction = (target, result) =>
                    {
                        if (target * result <= 0) return 0.5 - target * result;
                        else if (target * result > 1) return 0;
                        else return 0.5 * (1 - target * result) * (1 - target * result);
                    };
                    dNetworkLossFunction = (target, result) =>
                    {
                        if (target * result <= 0) return -result;
                        else if (target * result > 1) return 0;
                        else return -target * (1 - target * result);
                    };
                    break;
                case LossFunctionType.SquareLossClassification:
                    NetworkLossFunction = (target, result) => Math.Pow(1 - result * target, 2);
                    dNetworkLossFunction = (target, result) => -2 * target * (1 - target * result);
                    break;
                case LossFunctionType.LogisticLoss:
                    NetworkLossFunction = (target, result) => (1 / Math.Log(2)) * Math.Log(1 + Math.Exp(-target * result));
                    dNetworkLossFunction = (target, result) => (result * Math.Exp(result * target)) / (Math.Log(2) * Math.Exp(target * result) + Math.Log(2));
                    break;
                case LossFunctionType.ExponentialLoss:
                    NetworkLossFunction = (target, result) => Math.Exp(-target * result);
                    dNetworkLossFunction = (target, result) => -result * Math.Exp(-result * target);
                    break;
                case LossFunctionType.CrossEntropyLoss:
                    NetworkLossFunction = (target, result) => {
                        double t = (1 + result) / 2;
                        return -t * Math.Log(result) - (1 - t) * Math.Log(1 - result);
                    };
                    dNetworkLossFunction = (target, result) => {
                        double t = (1 + result) / 2;
                        return -((t - result) / ((1 - result) * result));
                    };
                    break;
            }
        }

        #region Layers

        internal void Add(Layer l)
        {
            Console.WriteLine($"Adding {l.Type}");
            switch (l.Type)
            {
                case Layer.LayerType.Convolutional:
                    Add(l as ConvolutionLayer);
                    break;
                case Layer.LayerType.FullyConnected:
                    Add(l as FullyConnectedLayer);
                    break;
                case Layer.LayerType.InputLayer:
                    Add(l as InputLayer);
                    break;
                case Layer.LayerType.Pooling:
                    Add(l as PoolingLayer);
                    break;
                case Layer.LayerType.ReLu:
                    Add(l as ReLuLayer);
                    break;
                case Layer.LayerType.SoftMax:
                    Add(l as SoftMaxLayer);
                    break;
                default:
                    throw new ArgumentException($"Unrecognized Layer-Type: {l.Type}");
            }
        }

        private void Add(FullyConnectedLayer l)
        {
            Layers.Push(l);
        }

        private void Add(InputLayer l)
        {
            InputLayer = l;
            Layers.Push(l);
        }

        private void Add(SoftMaxLayer l)
        {
            Layers.Push(l);
        }

        private void Add(ConvolutionLayer l)
        {
            Layers.Push(l);
        }

        private void Add(PoolingLayer l)
        {
            Layers.Push(l);
        }

        private void Add(ReLuLayer l)
        {
            Layers.Push(l);
        }

        public void AddInputLayer(int size)
        {
            Add(new InputLayer(size, Source));
        }

        public void AddFullyConnecterLayer(int size, ActivationFunctionType functionType = ActivationFunctionType.Sigmoid, bool scale = true)
        {
            Add(new FullyConnectedLayer(size, Layers.Peek(), Source, functionType, scale));
        }

        public void AddSoftMaxLayer()
        {
            Add(new SoftMaxLayer(Layers.Peek(), Source));
        }

        public void AddConvolutionLayer(int n_filters, int filtersize, int stride, int padding)
        {
            Add(new ConvolutionLayer(Layers.Peek(), n_filters, filtersize, stride, padding, Source));
        }

        public (int w, int h, int d) AddConvolutionLayer(int n_filters, int filtersize, int stride, int padding, (int, int, int) dims)
        {
            var cl = new ConvolutionLayer(Layers.Peek(), n_filters, filtersize, stride, padding, dims, Source);
            Add(cl);
            return cl.GetOutputSpatialExtent();
        }

        public (int, int, int) AddPoolingLayer(int tessellation)
        {
            var l = new PoolingLayer(Layers.Peek(), tessellation, Source);
            Add(l);
            return l.GetOutputSpatialExtent();

        }

        public (int, int, int) AddPoolingLayer(int tessellation, (int, int) dims)
        {
            var l = new PoolingLayer(Layers.Peek(), tessellation, dims, Source);
            Add(l);
            return l.GetOutputSpatialExtent();
        }

        public (int, int, int) AddPoolingLayer(int tessellation, (int, int, int) dims)
        {
            var l = new PoolingLayer(Layers.Peek(), tessellation, dims, Source);
            Add(l);
            return l.GetOutputSpatialExtent();
        }

        public void AddReLuLayer()
        {
            Add(new ReLuLayer(Layers.Peek(), Source));
        }

        #endregion

        public void SetInput(double[] inputs)
        {

#if DEBUG
            Debug.Assert(InputLayer.GetNeurons().Count == inputs.Length);
#endif
            InputLayer.AddInput(inputs);
        }

        internal void SetSource(NeuronSource s)
        {
            Source = s;
        }

        #region IO

        public void Print()
        {
            foreach (var layer in Layers.Reverse())
            {
                Console.WriteLine(layer.ToString());
                Console.WriteLine();
            }
        }

        public string SaveAsBinary(string filename)
        {
            Console.Clear();

            //string[] splitname = filename.Split('.');
            //for (int i = 1; System.IO.File.Exists(filename); i++)
            //{
            //    filename = $"{splitname[0]}{i}.{splitname.Skip(1).Aggregate((x, y) => $"{x}.{y}")}";
            //}
            Console.WriteLine($"Writing to file: {filename}");

            IO.NetworkToFile ntf = new IO.NetworkToFile(this);
            ntf.Write(IO.NetworkToFile.WriteMode.Binary, filename);

            return filename;
        }


        public static Network ReadFromFileBinary(string filename)
        {
            using (var fs = System.IO.File.OpenRead(filename))
            using (var br = new System.IO.BinaryReader(fs))
            {
                IO.BinaryNetworkReader networkReader = new IO.BinaryNetworkReader(br);
                var nn = networkReader.Read();
                return nn;
            }
        }

        #endregion

        internal Neuron FindNeuronByID(long ID)
        {
            return Source.GetGeneratedNeurons()[ID];
        }

        public double[] ForwardPass()
        {
            for (int i = Layers.Count - 1; i >= 0; i--)
            {
                Layers.ElementAt(i).EvaluateAllNeurons();
            }

            return Outputlayer.Outputs();
        }

        #region Learning

        long trainingExamplesSeen = 0;
        double Error;
        double TotalErrors;
        double SlidingWindowError;
        int SlidingWindowSize = 200;

        public void Backpropagate(double[] target)
        {
            if (Layers.Count < 2)
            {
                throw new IndexOutOfRangeException("Trying to backpropagate with 1 or fewer layers.");
            }
            var outputs = Outputlayer.Outputs();
            var zipped = target.Zip(outputs, (x, y) => NetworkLossFunction(x, y));
            double error = zipped.Sum();
            SlidingWindowError += error;
            TotalErrors += error;
            Console.Out.WriteLineAsync($"Examples: {trainingExamplesSeen++}");
            Console.Out.WriteLineAsync($"\tGuess : {Helper.StringyfyVector(Outputlayer.Outputs())}");
            Console.Out.WriteLineAsync($"\tActual: {Helper.StringyfyVector(target)}");
            Console.Out.WriteLineAsync($"\tERROR : {error}");
            Console.Out.WriteLineAsync($"\tAVGERR: {TotalErrors / trainingExamplesSeen}");
            if (trainingExamplesSeen % SlidingWindowSize == 0)
            {
                Console.Out.WriteLineAsync($"\tSLDERR: {SlidingWindowError / SlidingWindowSize}");
                SlidingWindowError = 0;
            }
            else
            {
                Console.WriteLine();
            }
            Console.WriteLine();
            Console.WriteLine();
            Error = error;
            Console.SetCursorPosition(0, Console.CursorTop - 8);

            Outputlayer.Backpropagate(targetvalues:target);
            for (int i = 1; i < Layers.Count; i++)
            {
                Layers.ElementAt(i).Backpropagate();
            }

            foreach (var layer in Layers)
            {
                Parallel.ForEach(layer.GetNeurons(), n => n.ResetDelta());
            }
        }

        #endregion

        #region Genetics
        
        public double[] GetGenome()
        {
            var genome = Source.GetGeneratedEdges().Select(e => e.Weight).ToList();
            foreach (var layer in Layers)
            {
                if(layer is ConvolutionLayer)
                {
                    var cl = layer as ConvolutionLayer;
                    for (int i = 0; i < cl.filters.Count; i++)
                    {
                        for (int d = 0; d < cl.filters[i].GetLength(0); d++)
                        {
                            for (int h = 0; h < cl.filters[i].GetLength(1); h++)
                            {
                                for (int w = 0; w < cl.filters[i].GetLength(2); w++)
                                {
                                    genome.Add(cl.filters[i][d, h, w]);
                                }
                            }
                        }
                    }
                    genome.AddRange(cl.biases);
                }
            }
            return genome.ToArray();
        }

        public void SetGenome(double[] genome)
        {
            var edges = Source.GetGeneratedEdges();
            for (int i = 0; i < edges.Count; i++)
            {
                edges[i].Weight = genome[i];
            }

        }

        public static void Mutate(List<Network> winners)
        {

        }
        #endregion
    }
}
