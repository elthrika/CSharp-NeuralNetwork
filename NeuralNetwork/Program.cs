using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;

namespace NeuralNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            //string s = TrainConv();
            //ReadConv(s);
            //TrainXOR();
            TrainMNIST();
            //TestMNISTFromSaved();
            Console.ReadKey();
        }

        static void ReadConv(string s)
        {
            
            Network nn = Network.ReadFromFileBinary(s);
            nn.Print();
        }

        static string TrainConv()
        {
            Network nn = new Network(Network.LossFunctionType.Logistical);
            nn.AddInputLayer(64*3);
            var ext = nn.AddConvolutionLayer(n_filters: 5, filtersize: 4, stride: 2, padding: 1, dims: (8, 8, 3));
            nn.AddPoolingLayer(2, dims: ext);
            nn.AddReLuLayer();
            nn.AddFullyConnecterLayer(10, ActivationFunctionType.Sigmoid);
            nn.AddSoftMaxLayer();
            nn.Print();
            for (int i = 0; i < 100; i++)
            {
                nn.SetInput(Helper.RandomVector(64 * 3));
                nn.ForwardPass();
                nn.Backpropagate(Helper.RandomVector(10));
            }
            return nn.SaveAsBinary("testNN.ann");
        }

        static void TrainXOR()
        {
            Network nn = new Network(Network.LossFunctionType.Logistical);
            nn.AddInputLayer(2);
            nn.AddFullyConnecterLayer(5, ActivationFunctionType.Sigmoid);
            nn.AddFullyConnecterLayer(2, ActivationFunctionType.Sigmoid);
            nn.AddSoftMaxLayer();
            nn.Print();

            Random random = new Random();

            double[] nullresult = new double[2] { 0.0, 1.0 };
            double[] oneresult  = new double[2] { 1.0, 0.0 };
            for (int i = 0; i < 30000; i++)
            {
                double[] input = new double[2] { random.Next(2), random.Next(2) };
                double[] result = (input[0] == 1) ^ (input[1] == 1) ? oneresult : nullresult;
                nn.SetInput(input);
                nn.ForwardPass();
                nn.Backpropagate(result);
            }

            int succ = 0;
            int tests = 3000;
            for (int i = 0; i < tests; i++)
            {
                double[] input = new double[2] { random.Next(2), random.Next(2) };
                double[] result = (input[0] == 1) ^ (input[1] == 1) ? oneresult : nullresult;
                nn.SetInput(input);
                double[] res = nn.ForwardPass();
                int maxidx = res.ToList().IndexOf(res.Max());
                double[] argmaxres = new double[res.Length];
                argmaxres[maxidx] = 1;
                bool eq = argmaxres.Zip(result, (x, y) => x == y).All(x => x);
                succ += eq.CompareTo(false);
                Console.ForegroundColor = eq ? ConsoleColor.Green : ConsoleColor.Red;
                Console.WriteLine($"{input[0]}^{input[1]} = {Helper.StringyfyVector(result)} - Guessed: {Helper.StringyfyVector(res)}");
            }
            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"{succ} / {tests} Correct");
        }

        static void TrainMNIST()
        {
            Network nn = new Network(Network.LossFunctionType.Logistical);
            nn.AddInputLayer(28 * 28 * 1);
            var ext = nn.AddConvolutionLayer(n_filters: 3, filtersize: 3, stride: 1, padding: 1, dims: (28, 28, 1));
            ext = nn.AddPoolingLayer(2, ext);
            //ext = nn.AddConvolutionLayer(n_filters: 2, filtersize: 3, stride: 1, padding: 1, dims: ext);
            nn.AddFullyConnecterLayer(10, ActivationFunctionType.Tanh);
            nn.AddSoftMaxLayer();
            nn.Print();

            foreach (var (l, i) in ReadMNISTData("train-labels.idx1-ubyte", "train-images.idx3-ubyte"))
            {
                nn.SetInput(i);
                double[] res = nn.ForwardPass();
                nn.Backpropagate(l);
            }
            Console.Clear();
            nn.Print();
            nn.SaveAsBinary("MNIST89.ann");

            int accurate = 0; int total = 0;
            foreach (var (l, i) in ReadMNISTData("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte"))
            {
                nn.SetInput(i);
                double[] res = nn.ForwardPass();
                int label = l.ToList().IndexOf(l.Max());
                int guess = res.ToList().IndexOf(res.Max());
                accurate += (label == guess).CompareTo(false);
                Console.ForegroundColor = label == guess ? ConsoleColor.Green : ConsoleColor.Red;
                Console.WriteLine($"Guess: {guess}\t{Helper.StringyfyVector(res)}\nActual:{label}\t{Helper.StringyfyVector(l)}");
                total++;
            }

            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"{accurate} / {total} correct");
        }

        static void TestMNISTFromSaved()
        {
            Network nn = Network.ReadFromFileBinary("MNIST89.ann");
            Console.ReadKey();
            nn.Print();
            Console.ReadKey();
            int accurate = 0; int total = 0;
            foreach (var (l, i) in ReadMNISTData("t10k-labels.idx1-ubyte", "t10k-images.idx3-ubyte"))
            {
                nn.SetInput(i);
                double[] res = nn.ForwardPass();
                int label = l.ToList().IndexOf(l.Max());
                int guess = res.ToList().IndexOf(res.Max());
                accurate += (label == guess).CompareTo(false);
                Console.ForegroundColor = label == guess ? ConsoleColor.Green : ConsoleColor.Red;
                Console.WriteLine($"Guess: {guess}\t{res.StringyfyVector()}\nActual:{label}\t{l.StringyfyVector()}");
                total++;
            }

            Console.ForegroundColor = ConsoleColor.White;
            Console.WriteLine($"{accurate} / {total} correct");
        }

        static IEnumerable<(double[], double[])> ReadMNISTData(string labelfile, string imagefile)
        {
            FileStream imagesfile = File.Open(imagefile, FileMode.Open);
            FileStream labelsfile = File.Open(labelfile, FileMode.Open);

            {
                int magic_number = ReadInt32(labelsfile);
                if (magic_number != 2049) throw new ArgumentException();
            }
            {
                int magic_number = ReadInt32(imagesfile);
                if (magic_number != 2051) throw new ArgumentException();
            }
            int lsize = ReadInt32(labelsfile);
            int isize = ReadInt32(imagesfile);
            System.Diagnostics.Debug.Assert(lsize == isize);
            int rows = ReadInt32(imagesfile);
            int cols = ReadInt32(imagesfile);

            for (int i = 0; i < lsize; i++)
            {
                double[] label = GetNextLabel(labelsfile);
                double[] image = ReadImageFile(imagesfile, rows, cols);
                yield return (label, image);
            }
        }

        static double[] GetNextLabel(FileStream fs)
        {
            double[] labels = new double[10];
            
            labels[fs.ReadByte()] = 1;

            return labels;
        }

        static double[] ReadImageFile(FileStream fs, int rows, int cols)
        {
            double[] images = new double[rows*cols];

            for (int r = 0; r < rows * cols; r++)
            {
                images[r] = fs.ReadByte()/255.0D;
            }

            return images;
        }

        static int ReadInt32(FileStream fs)
        {
            byte[] bs = new byte[4];
            fs.Read(bs, 0, 4);
            
            return BitConverter.ToInt32(bs.Reverse().ToArray(), 0);
        }

        static void DrawMNIST(double[] image)
        {
            int imagedim = (int)Math.Sqrt(image.Length);
            for (int j = 0; j < image.Length; j++)
            {
                if (image[j] > 0)
                    Console.Write("\u2588");
                else
                    Console.Write("\u2592");
                if (j % imagedim == imagedim - 1)
                    Console.WriteLine();
            }
        }
    }
}
