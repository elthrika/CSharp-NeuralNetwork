using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;

namespace NeuralNetwork
{
    internal class ConvolutionLayer : Layer
    {
        int SizeW, SizeH, SizeD;
        internal List<double[,,]> filters;
        internal double[] biases;
        Neuron[,,] extendedInputs;

        double[,,] InputValues;
        double[,,] DeltaXs;
        Dictionary<long, double> NeuronDeltas;

        internal int N_filters { get; private set; }
        internal int Filtersize { get; private set; }
        internal int Stride { get; private set; }
        internal int Padding { get; private set; }
        internal int InputWidth { get; private set; }
        internal int InputHeight { get; private set; }
        internal int InputDepth { get; private set; }

        #region INIT

        internal ConvolutionLayer(Layer previouslayer, int n_filters, int filtersize, int stride, int padding, (int width, int heigth, int depth) inputVolumeDims, Network n)
            : base(LayerType.Convolutional, n)
        {
            Init(previouslayer, n_filters, filtersize, stride, padding, inputVolumeDims);
        }

        internal ConvolutionLayer(Layer previouslayer, int n_filters, int filtersize, int stride, int padding, (int width, int heigth) inputVolumeDims, Network n)
            : base(LayerType.Convolutional, n)
        {
            Init(previouslayer, n_filters, filtersize, stride, padding, (inputVolumeDims.width, inputVolumeDims.heigth, 1));
        }

        internal ConvolutionLayer(Layer previouslayer, int n_filters, int filtersize, int stride, int padding, Network n)
            : base(LayerType.Convolutional, n)
        {
            // if no dimensions are given, assume a square, two dimensional (x, x, 1) input volume
            int sidelength = (int)Math.Sqrt(previouslayer.Size);
            Init(previouslayer, n_filters, filtersize, stride, padding, (sidelength, sidelength, 1));
        }

        internal (int w, int h, int d) GetOutputSpatialExtent()
        {
            return (SizeW, SizeH, SizeD);
        } 

        private void Init(Layer previouslayer, int n_filters, int filtersize, int stride, int padding, (int width, int height, int depth) inputVolumeDims)
        {
            N_filters = n_filters;
            Filtersize = filtersize;
            Padding = padding;
            Stride = stride;
            InputWidth = inputVolumeDims.width;
            InputHeight = inputVolumeDims.height;
            InputDepth = inputVolumeDims.depth;

            if (n_filters < 1 || filtersize < 1 || stride < 1) throw new ArgumentException($"n_filters[{n_filters}], filtersize[{filtersize}], stride[{stride}] cannot be <0");
            if ((InputWidth - filtersize + 2 * padding) % stride != 0) Console.WriteLine($"Cannot do that {(InputWidth - filtersize + 2 * padding)} % {stride} = {(InputWidth - filtersize + 2 * padding) % stride}");
            if ((InputHeight - filtersize + 2 * padding) % stride != 0) Console.WriteLine($"Cannot do that {(InputHeight - filtersize + 2 * padding)} % {stride} = {(InputHeight - filtersize + 2 * padding) % stride}");

            SizeW = (InputWidth - filtersize + 2 * padding) / stride + 1;
            SizeH = (InputHeight - filtersize + 2 * padding) / stride + 1;
            SizeD = n_filters;
            Size =  SizeW * SizeH * SizeD;
            AllNeurons = new Neuron[Size];

            filters = new List<double[,,]>(n_filters);
            biases  = new double[n_filters];
            for (int i = 0; i < n_filters; i++)
            {
                biases[i] = Helper.Random;
                filters.Add(Helper.RandomMatrix(filtersize, filtersize, InputDepth));
            }

            for (int i = 0; i < Size; i++)
            {
                AllNeurons[i] = Source.GetConvolutionNeuron(this);
            }

            NeuronDeltas = new Dictionary<long, double>(InputDepth * InputHeight * InputWidth);
            
            extendedInputs = new Neuron[InputDepth, InputWidth, InputHeight];
            int c = 0;
            var previousNeurons = previouslayer.GetNeurons();
            for (int d = 0; d < InputDepth; d++)
            {
                for (int h = 0; h < InputHeight; h++)
                {
                    for (int w = 0; w < InputWidth; w++)
                    {
                        extendedInputs[d, h, w] = previousNeurons[c++];
                    }
                }
            }
            extendedInputs = Helper.AddXYPadding(extendedInputs, padding, () => Source.GetBiasNeuron(this, 0));

            MakeEdgesToPreviousLayer(previouslayer);
        }

        private void MakeEdgesToPreviousLayer(Layer previousLayer)
        {
            int depth = extendedInputs.GetLength(0);
            int height = extendedInputs.GetLength(1);
            int width = extendedInputs.GetLength(2);
            for (int h = 0; h < height - Filtersize + 1; h += Stride)
            {
                for (int w = 0; w < width - Filtersize + 1; w += Stride)
                {
                    for (int d = 0; d < depth; d++)
                    {
                        for (int i = 0; i < Filtersize; i++)
                        {
                            for (int j = 0; j < Filtersize; j++)
                            {
                                var previous_neuron = extendedInputs[d, h + j, w + i];
                                for (int f = 0; f < N_filters; f++)
                                {
                                    var my_neuron = GetNeuron(f, h / Stride, w / Stride);
                                    Edge e = Source.MakeEdge(1, previous_neuron, my_neuron);
                                }
                            }
                        }
                    }
                }
            }
        }

        #endregion

        private void SetNeuron(int d, int h, int w, double val)
        {
            if (d >= SizeD) throw new ArgumentException($"Given Depth {d} is >= {SizeD}");
            if (h >= SizeH) throw new ArgumentException($"Given Height {h} is >= {SizeH}");
            if (w >= SizeW) throw new ArgumentException($"Given Width {w} is >= {SizeW}");
            AllNeurons[w + (h * SizeW) + (d * SizeW * SizeH)].SetValue(val);
        }

        private Neuron GetNeuron(int d, int h, int w)
        {
            if (d >= SizeD) throw new ArgumentException($"Given Depth {d} is >= {SizeD}");
            if (h >= SizeH) throw new ArgumentException($"Given Height {h} is >= {SizeH}");
            if (w >= SizeW) throw new ArgumentException($"Given Width {w} is >= {SizeW}");
            return AllNeurons[w + (h * SizeW) + (d * SizeW * SizeH)];
        }

        internal override void EvaluateAllNeurons()
        {
            int sizeDivDepth = Size / N_filters;

            int extdWidth = InputWidth + 2 * Padding;
            int extdHeight = InputHeight + 2 * Padding;
            InputValues = new double[InputDepth, extdHeight, extdHeight];

            Parallel.For(0, InputDepth, d =>
            {
                for (int h = 0; h < extdHeight; h++)
                {
                    for (int w = 0; w < extdWidth; w++)
                    {
                        InputValues[d, h, w] = extendedInputs[d, h, w].Value;
                    }
                }
            });

            Parallel.For(0, N_filters, d =>
            {
                double[,] conv = Convolute(InputValues, filters[d], Stride, Filtersize, biases[d]);
                for (int h = 0; h < conv.GetLength(0); h++)
                {
                    for (int w = 0; w < conv.GetLength(1); w++)
                    {
                        SetNeuron(d, h, w, conv[h, w]);
                    }
                }
            });
        }

        #region Convolutions
        private double[,] Convolute(double[,,] arr, double[,,] filter, int stride, int filtersize, double bias = 0)
        {
            return Convolute(arr, filter, stride, (filtersize, filtersize), bias);
        }

        private double[,] Convolute(double[,,] arr, double[,,] filter, int stride, (int h, int w) filtersize, double bias = 0)
        {
            if (arr.GetLength(0) != filter.GetLength(0)) throw new ArgumentException("Depth of filter and input do not agree");
            if (filter.GetLength(1) != filtersize.h || filter.GetLength(2) != filtersize.w) throw new ArgumentException("given filtersize is not actual filtersize");

            int depth = arr.GetLength(0);
            int height = arr.GetLength(1);
            int width = arr.GetLength(2);

            double[,] rval = new double[(height - filtersize.h) / stride + 1, (width - filtersize.w) / stride + 1];
            for (int h = 0; h < height - filtersize.h + 1; h += stride)
            {
                for (int w = 0; w < width - filtersize.w + 1; w += stride)
                {
                    double sum = bias;
                    for (int d = 0; d < depth; d++)
                    {
                        for (int i = 0; i < filtersize.h; i++)
                        {
                            for (int j = 0; j < filtersize.w; j++)
                            {
                                sum += arr[d, h + i, w + j] * filter[d, i, j];
                            }
                        }
                    }
                    rval[h / stride, w / stride] = sum;
                }
            }
            return rval;
        }

        private double[,,] Convolute2(double[,,] arr, double[,] filter, int stride, int filtersize, double bias = 0)
        {
            return Convolute2(arr, filter, stride, (filtersize, filtersize), bias);
        }


        private double[,,] Convolute2(double[,,] arr, double[,] filter, int stride, (int h, int w) filtersize, double bias = 0)
        {
            //if (arr.GetLength(0) != filter.GetLength(0)) throw new ArgumentException("Depth of filter and input do not agree");
            if (filter.GetLength(0) != filtersize.h || filter.GetLength(1) != filtersize.w) throw new ArgumentException("given filtersize is not actual filtersize");

            int depth = arr.GetLength(0);
            int height = arr.GetLength(1);
            int width = arr.GetLength(2);

            double[,,] rval = new double[depth, (height - filtersize.h) / stride + 1, (width - filtersize.w) / stride + 1];
            for (int h = 0; h < height - filtersize.h + 1; h += stride)
            {
                for (int w = 0; w < width - filtersize.w + 1; w += stride)
                {
                    for (int d = 0; d < depth; d++)
                    {
                        double sum = bias;
                        for (int i = 0; i < filtersize.h; i++)
                        {
                            for (int j = 0; j < filtersize.w; j++)
                            {
                                sum += arr[d, h + i, w + j] * filter[i, j];
                            }
                        }
                        rval[d, h / stride, w / stride] = sum;
                    }
                }
            }
            return rval;
        }
        #endregion

        internal double GetDeltaForNeuron(Neuron n)
        {
            return NeuronDeltas[n.ID];
        }

        internal override void Backpropagate(double[] targetvalues = null)
        {
            double[,,] deltaOs = new double[SizeD, SizeH, SizeW];
            double[,,] Xs = InputValues;
            for (int i = 0; i < AllNeurons.Length; i++)
            {
                double targetvalue = targetvalues == null ? 0 : targetvalues[i];
                AllNeurons[i].Backpropagate(targetvalue);
                deltaOs[i / (SizeW * SizeH), (i / SizeW) % SizeH, i % SizeW] = (AllNeurons[i] as ConvolutionNeuron).DeltaO;
            }

            // calculate Filter Delta
            int paddingXs = ((Filtersize - 1) * Stride + SizeW - InputWidth) / 2;
            if(paddingXs - Padding != 0)
            {
                Xs = Helper.AddXYPadding(InputValues, paddingXs - Padding);
            }
            for (int i = 0; i < N_filters; i++)
            {
                var dim = deltaOs.ExtractDim(0, i);
                var deltaF = Convolute2(Xs, dim, Stride, (SizeH, SizeW));
                filters[i] = filters[i].AddMatrix(deltaF);
                biases[i] += dim.Cast<double>().Sum();
            }

            //calculate input deltas
            int paddingOs = ((InputWidth - 1) * Stride + Filtersize - SizeW) / 2;
            if (((InputWidth - 1) * Stride + Filtersize - SizeW) % 2 != 0) Console.WriteLine($"Cannot do that {(InputWidth - 1) * Stride + Filtersize - SizeW} % {2} = {((InputWidth - 1) * Stride + Filtersize - SizeW) % 2}");

            deltaOs = Helper.AddXYPadding(deltaOs, paddingOs);
            DeltaXs = new double[InputDepth, InputHeight, InputWidth];
            for (int i = 0; i < InputDepth; i++)
            {
                double[,] deltaXi = Convolute(deltaOs, filters.ExtractN(i).Transpose3D((0, 2, 1)), Stride, Filtersize);
                DeltaXs = DeltaXs.AddMatrix(deltaXi, i);
            }

            for (int d = 0; d < InputDepth; d++)
            {
                for (int h = Padding; h < InputHeight + Padding; h++)
                {
                    for (int w = Padding; w < InputWidth + Padding; w++)
                    {
                        NeuronDeltas[extendedInputs[d, h, w].ID] = DeltaXs[d, h - Padding, w - Padding];
                    }
                }
            }

        }

        #region IO

        public override string ToString()
        {
            var s = $"Convolutional Layer\n{InputWidth}x{InputHeight}x{InputDepth} => {GetOutputSpatialExtent()}\n";
            s += $"Stride: {Stride}, Padding {Padding}\n{N_filters} Filters";
            return s;
        }

        #endregion
    }
}
