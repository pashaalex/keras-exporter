using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace MyModel
{
    class Program
    {

        static void Main(string[] args)
        {
            //ResNet50
            {
                Console.WriteLine("ResNet50...");
                var net = new ResNet50("ResNet50.dat");
                float[,,] img = NetUtils.PrepareImageResNet("test_dog.png");
                Stopwatch time_measure = new Stopwatch();
                time_measure.Start();
                float[] prediction = net.Process(img);
                time_measure.Stop();
                Console.WriteLine("Time: " + (time_measure.ElapsedMilliseconds / 1000.0).ToString("0.000") + " s");
                Console.WriteLine("Top 3 results: " + string.Join(", ", NetUtils.DecodeImageNetResult(prediction, 3)));
                Console.WriteLine("--------------\n");
            }

            //InceptionV3
            {
                Console.WriteLine("InceptionV3...");
                var net = new InceptionV3("InceptionV3.dat");
                float[,,] img = NetUtils.PrepareImageInceptionV3("test_dog.png");
                Stopwatch time_measure = new Stopwatch();
                time_measure.Start();
                float[] prediction = net.Process(img);
                time_measure.Stop();
                Console.WriteLine("Time: " + (time_measure.ElapsedMilliseconds / 1000.0).ToString("0.000") + " s");
                Console.WriteLine("Top 3 results: " + string.Join(", ", NetUtils.DecodeImageNetResult(prediction, 3)));
                Console.WriteLine("--------------\n");
            }

            //MobileNet
            {
                Console.WriteLine("MobileNet...");
                var net = new MobileNet("MobileNet.dat");
                float[,,] img = NetUtils.PrepareImageMobileNet("test_dog.png");
                Stopwatch time_measure = new Stopwatch();
                time_measure.Start();
                float[] prediction = net.Process(img);
                time_measure.Stop();
                Console.WriteLine("Time: " + (time_measure.ElapsedMilliseconds / 1000.0).ToString("0.000") + " s");
                Console.WriteLine("Top 3 results: " + string.Join(", ", NetUtils.DecodeImageNetResult(prediction, 3)));
                Console.WriteLine("--------------\n");
            }

            // Xception
            {
                Console.WriteLine("Xception...");
                var net = new Xception("Xception.dat");
                float[,,] img = NetUtils.PrepareImageInceptionV3("test_dog.png");
                Stopwatch time_measure = new Stopwatch();
                time_measure.Start();
                float[] prediction = net.Process(img);
                time_measure.Stop();
                Console.WriteLine("Time: " + (time_measure.ElapsedMilliseconds / 1000.0).ToString("0.000") + " s");
                Console.WriteLine("Top 3 results: " + string.Join(", ", NetUtils.DecodeImageNetResult(prediction, 3)));
                Console.WriteLine("--------------\n");
            }


            Console.WriteLine("Press a key");
            Console.ReadKey();
        }
    }
}
