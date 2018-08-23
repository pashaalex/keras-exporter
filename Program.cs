using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;

namespace MyModel
{
    class Program
    {
        public static float[,,] PrepareImageResNet(string fName)
        {
            int W = 224;
            int H = 224;
            using (Bitmap bmp_src = (Bitmap)Bitmap.FromFile(fName))
            using (Bitmap bmp = new Bitmap(bmp_src, 224, 224))
            {
                float[,,] src = NetUtils.PrepareImageRGB(bmp);
                for (int Y = 0; Y < H; Y++)
                {
                    for (int X = 0; X < W; X++)
                    {
                        src[Y, X, 0] = src[Y, X, 0] - 103.939F;
                        src[Y, X, 1] = src[Y, X, 1] - 116.779F;
                        src[Y, X, 2] = src[Y, X, 2] - 123.68F;
                    }
                }
                return src;
            }
        }

        public static float[,,] PrepareImageInceptionV3(string fName)
        {
            int W = 299;
            int H = 299;
            using (Bitmap bmp_src = (Bitmap)Bitmap.FromFile(fName))
            using (Bitmap bmp = new Bitmap(bmp_src, W, H))
            {
                float[,,] src = NetUtils.PrepareImageRGB(bmp);
                for (int Y = 0; Y < H; Y++)
                {
                    for (int X = 0; X < W; X++)
                    {
                        src[Y, X, 0] = (src[Y, X, 2] / 127.5F) -1F;
                        src[Y, X, 1] = (src[Y, X, 1] / 127.5F) -1F;
                        src[Y, X, 2] = (src[Y, X, 0] / 127.5F) -1F;
                    }
                }
                return src;
            }
        }

        public static float[,,] PrepareImageMobileNet(string fName)
        {
            int W = 224;
            int H = 224;
            using (Bitmap bmp_src = (Bitmap)Bitmap.FromFile(fName))
            using (Bitmap bmp = new Bitmap(bmp_src, W, H))
            {
                float[,,] src = NetUtils.PrepareImageRGB(bmp);
                for (int Y = 0; Y < H; Y++)
                {
                    for (int X = 0; X < W; X++)
                    {
                        src[Y, X, 0] = (src[Y, X, 0] / 127.5F) - 1F;
                        src[Y, X, 1] = (src[Y, X, 1] / 127.5F) - 1F;
                        src[Y, X, 2] = (src[Y, X, 2] / 127.5F) - 1F;
                    }
                }
                return src;
            }
        }

        static void Main(string[] args)
        {
            //ResNet50
            {
                Console.WriteLine("ResNet50...");
                var net = new ResNet50("ResNet50.dat");
                float[,,] img = PrepareImageResNet("test_dog.png");
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
                float[,,] img = PrepareImageInceptionV3("test_dog.png");
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
                float[,,] img = PrepareImageMobileNet("test_dog.png");
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
                float[,,] img = PrepareImageInceptionV3("test_dog.png");
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
