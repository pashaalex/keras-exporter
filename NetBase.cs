using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace MyModel
{
    public class NetBase
    {
        protected Dictionary<string, Array> Weights = new Dictionary<string, Array>();

        #region Utils
        public Array ReadTensor(BinaryReader br)
        {
            int dim = br.ReadInt32();
            int[] dims = new int[dim];
            int[] ind = new int[dim];
            for (int i = 0; i < dim; i++)
            {
                dims[i] = br.ReadInt32();
                ind[i] = 0;
            }

            Array arr = Array.CreateInstance(typeof(float), dims, ind);

            while (true)
            {
                float f = br.ReadSingle();
                arr.SetValue(f, ind);
                int i = dim - 1;
                while (i >= 0)
                {
                    ind[i]++;
                    if (ind[i] == dims[i])
                    {
                        ind[i] = 0;
                        i--;
                    }
                    else
                        break;
                }
                if (i == -1) break;
            }
            return arr;
        }
        #endregion

        #region Activation function
        protected void SoftMax1D(float[] src)
        {
            int W = src.Length;

            double sum = 0;
            for (int i = 0; i < W; i++)
                sum += (float)Math.Exp(src[i]);

            for (int i = 0; i < W; i++)
                src[i] = (float)(Math.Exp(src[i]) / sum);
        }

        protected void SoftMax3D(float[,,] src)
        {
            double sum = 0;
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    for (int k = 0; k < src.GetLength(2); k++)
                        sum += (float)Math.Exp(src[i, j, k]);

            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    for (int k = 0; k < src.GetLength(2); k++)
                        src[i, j, k] = (float)(Math.Exp(src[i, j, k]) / sum);
        }



        protected void ReLu6_1D(float[] src)
        {
            for (int i = 0; i < src.Length; i++)
                src[i] = Math.Min(Math.Max(src[i], 0), 6);
        }

        protected void ReLu6_2D(float[,] src)
        {
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    src[i, j] = Math.Min(Math.Max(src[i, j], 0), 6);
        }
        protected void ReLu6_3D(float[,,] src)
        {
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    for (int k = 0; k < src.GetLength(2); k++)
                        src[i, j, k] = Math.Min(Math.Max(src[i, j, k], 0), 6);
        }

        protected void ReLu1D(float[] src)
        {
            for (int i = 0; i < src.Length; i++)
                if (src[i] < 0) src[i] = 0;
        }

        protected void ReLu2D(float[,] src)
        {
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    if (src[i,j] < 0) src[i,j] = 0;
        }

        protected void ReLu3D(float[,,] src)
        {
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    for (int k = 0; k < src.GetLength(2); k++)
                        if (src[i, j, k] < 0) src[i, j, k] = 0;
        }

        protected void Sigmoid1D(float[] src)
        {
            for (int i = 0; i < src.Length; i++)
                src[i] = (float)(1.0 / (1.0 + Math.Exp(-src[i])));
        }

        protected void Sigmoid2D(float[,] src)
        {
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    src[i, j] = (float)(1.0 / (1.0 + Math.Exp(-src[i, j])));
        }

        protected void Sigmoid3D(float[,,] src)
        {
            for (int i = 0; i < src.GetLength(0); i++)
                for (int j = 0; j < src.GetLength(1); j++)
                    for (int k = 0; k < src.GetLength(2); k++)
                        src[i, j, k] = (float)(1.0 / (1.0 + Math.Exp(-src[i, j, k])));
        }
        #endregion

        #region ReShape

        protected Array Reshape(Array src, params int[] dims)
        {
            int[] ind = new int[dims.Length];
            Array arr = Array.CreateInstance(typeof(float), dims, ind);

            foreach (float f in src)
            {
                arr.SetValue(f, ind);
                int p = ind.Length - 1;
                while (p >= 0)
                {
                    ind[p]++;
                    if (ind[p] == dims[p])
                    {
                        ind[p] = 0;
                        p = p - 1;
                    }
                    else
                        break;
                }
            }

            //Array.Copy(src, arr, src.Length);
            return arr;
        }

        protected float[] GlobalAveragePooling2D(float[,,] src)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            float[] res = new float[Deep];
            for (int f = 0; f < Deep; f++)
            {
                float sum = 0;
                float cnt = 0;
                for (int x = 0; x < W; x++)
                    for (int y = 0; y < H; y++)
                    {
                        sum += src[x, y, f];
                        cnt += 1;
                    }
                res[f] = sum / cnt;
            }
            return res;
        }

        protected float[] Flatten(float[,,] src)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            float[] res = new float[W * H * Deep];
            int i = 0;
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                    for (int f = 0; f < Deep; f++)
                        res[i++] = src[x, y, f];
            return res;
        }

        protected float[,,] Concatenate3D(params Array[] arr_in)
        {
            List<float[,,]> arr = arr_in.Select(n => (float[,,])n).ToList();            
            int W = arr[0].GetLength(0);
            int H = arr[0].GetLength(1);
            int l = 0;

            foreach (float[,,] t in arr)
            {
                if ((t.GetLength(0) != W) || (t.GetLength(1) != H))
                    throw new Exception("Dimension mismatch");
                l += t.GetLength(2);
            }

            float[,,] res = new float[W, H, l];
            l = 0;
            foreach (float[,,] t in arr)
            {
                int Da = t.GetLength(2);
                for (int d = 0; d < Da; d++)
                    for (int x = 0; x < W; x++)
                        for (int y = 0; y < H; y++)
                            res[x, y, d + l] = t[x, y, d];
                l += Da;
            }

            return res;
        }

        protected float[,,] Add3D(float[,,] a, float[,,] b)
        {
            if (a.GetLength(0) != b.GetLength(0)) throw new Exception("Dimension mismatch");
            if (a.GetLength(1) != b.GetLength(1)) throw new Exception("Dimension mismatch");
            if (a.GetLength(2) != b.GetLength(2)) throw new Exception("Dimension mismatch");

            int W = a.GetLength(0);
            int H = a.GetLength(1);
            int Da = a.GetLength(2);

            float[,,] res = new float[W, H, Da];
            for (int d = 0; d < Da; d++)
                for (int x = 0; x < W; x++)
                    for (int y = 0; y < H; y++)
                        res[x, y, d] = a[x, y, d] + b[x, y, d];

            return res;
        }

        #endregion

        protected float[,,] BatchNormalization3D(float[,,] src, float[] gamma, float[] betta, float[] mean, float[] std, float epsilon, float mean_)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            float[,,] res = new float[W, H, Deep];

            for (int c = 0; c < Deep; ++c)
            {
                float scale = gamma[c] / (float)Math.Sqrt(std[c] + epsilon);
                float offset = betta[c] - scale * mean[c];

                for (int x = 0; x < W; x++)
                    for (int y = 0; y < H; y++)
                        res[x, y, c] = scale * src[x, y, c] + offset;
            }

            return res;
        }

        protected float[,,] ZeroPadding2D(float[,,] src, int top, int bottom, int left, int right)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            int resW = W + left + right;
            int resH = H + top + bottom;

            float[,,] res = new float[resW, resH, Deep];
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                    for (int j = 0; j < Deep; j++)
                        res[left + x, top + y, j] = src[x, y, j];
            return res;
        }

        protected float[,,] Conv2d(float[,,,] kernel, float[] bias, float[,,] src, bool IsPaddingSame = false, int strideX = 1, int strideY = 1)
        {
            // Kernel is 4D: X, Y, Chanel, FilterCount
            int kernelW = kernel.GetLength(0);
            int kernelH = kernel.GetLength(1);
            int filterChannel = kernel.GetLength(2);
            int filterCount = kernel.GetLength(3);

            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            if (filterChannel != Deep) throw new Exception("Channel count mismatch");

            int resW = 0;
            int resH = 0;
            float[,,] res = null;

            if (!IsPaddingSame)
            {                
                resW = 1 + (W - kernelW) / strideX;
                resH = 1 + (H - kernelH) / strideY;
                res = new float[resW, resH, filterCount];

                //for (int x = 0; x < resW; x++)
                Parallel.For(0, resW, x =>
                {
                    for (int y = 0; y < resH; y++)
                        for (int f = 0; f < filterCount; f++)
                        {
                            res[x, y, f] = bias[f];
                            for (int d = 0; d < Deep; d++)
                                for (int kx = 0; kx < kernelW; kx++)
                                    for (int ky = 0; ky < kernelH; ky++)
                                        res[x, y, f] += src[x * strideX + kx, y * strideY + ky, d] * kernel[kx, ky, d, f];
                        }
                });
            }
            else
            {
                resW = (int)Math.Ceiling(W / (float)strideX);
                resH = (int)Math.Ceiling(H / (float)strideY);
                res = new float[resW, resH, filterCount];

                int dkw = (kernelW - 1) / 2;
                int dkh = (kernelH - 1) / 2;

                //for (int x = 0; x < resW; x++)
                Parallel.For(0, resW, x =>
                {
                    for (int y = 0; y < resH; y++)
                        for (int f = 0; f < filterCount; f++)
                        {
                            res[x, y, f] = bias[f];
                            for (int d = 0; d < Deep; d++)
                                for (int kx = 0; kx < kernelW; kx++)
                                    for (int ky = 0; ky < kernelH; ky++)
                                        if ((x + kx - dkw >= 0) && (y + ky - dkh >= 0) &&
                                            (x + kx - dkw < W) && (y + ky - dkh < H))
                                            res[x, y, f] += src[x * strideX + kx - dkw, y * strideY + ky - dkh, d] * kernel[kx, ky, d, f];
                        }
                });
            }

            return res;
        }

        protected float[,,] DepthwiseConv2D(float[,,,] kernel, float[] bias, float[,,] src, bool IsPaddingSame = false, int strideX = 1, int strideY = 1)
        {
            // Kernel is 4D: X, Y, Chanel, Multiplier
            int kernelW = kernel.GetLength(0);
            int kernelH = kernel.GetLength(1);
            int filterCount = kernel.GetLength(2);
            int filterMultiplier = kernel.GetLength(3);

            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            if (filterCount != Deep) throw new Exception("Channel count mismatch");

            int resW = 0;
            int resH = 0;
            float[,,] res = null;

            if (!IsPaddingSame)
            {
                resW = 1 + (W - kernelW) / strideX;
                resH = 1 + (H - kernelH) / strideY;
                res = new float[resW, resH, filterCount * filterMultiplier];

                //for (int x = 0; x < resW; x++)
                Parallel.For(0, resW, x =>
                {
                    for (int y = 0; y < resH; y++)
                        for (int f = 0; f < filterCount; f++)
                            for (int fm = 0; fm < filterMultiplier; fm++)
                            {
                                res[x, y, f + fm * filterCount] = bias[fm];
                                //for (int d = 0; d < Deep; d++)
                                    for (int kx = 0; kx < kernelW; kx++)
                                        for (int ky = 0; ky < kernelH; ky++)
                                            res[x, y, f + fm * filterCount] += src[x * strideX + kx, y * strideY + ky, f] * kernel[kx, ky, f, fm];
                            }
                });
            }
            else
            {
                resW = W / strideX;
                resH = H / strideY;
                res = new float[resW, resH, filterCount];

                int dkw = (kernelW - 1) / 2;
                int dkh = (kernelH - 1) / 2;

                //for (int x = 0; x < resW; x++)
                Parallel.For(0, resW, x =>
                {
                    for (int y = 0; y < resH; y++)
                        for (int f = 0; f < filterCount; f++)
                            for (int fm = 0; fm < filterMultiplier; fm++)
                            {
                                res[x, y, f + fm * filterCount] = bias[fm];
                                //for (int d = 0; d < Deep; d++)
                                    for (int kx = 0; kx < kernelW; kx++)
                                        for (int ky = 0; ky < kernelH; ky++)
                                            if ((x + kx - dkw >= 0) && (y + ky - dkh >= 0) &&
                                                (x + kx - dkw < W) && (y + ky - dkh < H))
                                                res[x, y, f + fm * filterCount] += src[x * strideX + kx - dkw, y * strideY + ky - dkh, f] * kernel[kx, ky, f, fm];
                            }
                });
            }

            return res;
        }

        protected float[,,] SeparableConv2D(float[,,,] kernel1, float[,,,] kernel2, float[] bias, float[,,] src, bool IsPaddingSame = false, int strideX = 1, int strideY = 1)
        {
            int kernelW = kernel1.GetLength(0);
            int kernelH = kernel1.GetLength(1);
            int filterCount = kernel1.GetLength(2);
            int filterMultiplier = kernel1.GetLength(3);

            float[,,] res1 = DepthwiseConv2D(kernel1, new float[filterMultiplier], src, IsPaddingSame, strideX, strideY);
            float[,,] res2 = Conv2d(kernel2, bias, res1, IsPaddingSame, strideX, strideY);

            return res2;
        }



        protected float[,,] AveragePooling2D(int poolX, int poolY, float[,,] src, int strideX = 1, int strideY = 1, bool IsSamePadding = false)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            if (!IsSamePadding)
            {
                //int resW = W / strideX;
                //int resH = H / strideY;
                int resW = 1 + (W - poolX) / strideX;
                int resH = 1 + (H - poolX) / strideY;
                float[,,] res = new float[resW, resH, Deep];
                for (int d = 0; d < Deep; d++)
                    for (int x = 0; x < resW; x++)
                        for (int y = 0; y < resH; y++)
                        {
                            float sum = 0;
                            float cnt = 0;
                            for (int i = x * strideX; i < x * strideX + poolX; i++)
                                for (int j = y * strideX; j < y * strideX + poolY; j++)
                                {
                                    sum += src[i, j, d];
                                    cnt += 1;
                                }
                            res[x, y, d] = sum / cnt;
                        }
                return res;
            }
            else
            {
                int resW = (int)Math.Ceiling(W / (float)strideX);
                int resH = (int)Math.Ceiling(H / (float)strideY);
                float[,,] res = new float[resW, resH, Deep];

                //int dkw = (poolX - 1) / 2;
                //int dkh = (poolY - 1) / 2;

                int dkw = W % strideX;
                int dkh = H % strideY;

                for (int x = 0; x < resW; x++)
                    for (int y = 0; y < resH; y++)
                        for (int f = 0; f < Deep; f++)
                        {
                            float sum = 0;
                            float cnt = 0;
                            for (int kx = 0; kx < poolX; kx++)
                                for (int ky = 0; ky < poolY; ky++)
                                {
                                    int cX = x * strideX + kx - dkw;
                                    int cY = y * strideY + ky - dkh;
                                    if ((cX >= 0) && (cY >= 0) && (cX < W) && (cY < H)) sum += src[cX, cY, f];
                                    cnt += 1;
                                }
                            res[x, y, f] = sum / cnt;
                        }
                
                return res;
            }

            /*
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            int resW = W / kw;
            int resH = H / kh;
            float[,,] res = new float[resW, resH, Deep];
            for (int d = 0; d < Deep; d++)
                for (int x = 0; x < resW; x++)
                    for (int y = 0; y < resH; y++)
                    {
                        float m = 0;
                        float cnt = 0;
                        for (int i = x * kw; i < (x + 1) * kw; i++)
                            for (int j = y * kh; j < (y + 1) * kh; j++)
                            {
                                m += src[i, j, d];
                                cnt += 1.0F;
                            }                                
                        res[x, y, d] = m / cnt;
                    }
            return res;*/
        }
        
        protected float[,,] MaxPool2d(int poolX, int poolY, int strideX, int strideY, float[,,] src, bool IsSamePadding = false)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            if (!IsSamePadding)
            {
                //int resW = W / strideX;
                //int resH = H / strideY;
                int resW = 1 + (W - poolX) / strideX;
                int resH = 1 + (H - poolX) / strideY;
                float[,,] res = new float[resW, resH, Deep];
                for (int d = 0; d < Deep; d++)
                    for (int x = 0; x < resW; x++)
                        for (int y = 0; y < resH; y++)
                        {
                            float max = float.MinValue;                            
                            for (int i = x * strideX; i < x * strideX + poolX; i++)
                                for (int j = y * strideX; j < y * strideX + poolY; j++)
                                    if (src[i, j, d] > max)
                                        max = src[i, j, d];                                
                            res[x, y, d] = max;
                        }
                return res;
            }
            else
            {
                int resW = (int)Math.Ceiling(W / (float)strideX);
                int resH = (int)Math.Ceiling(H / (float)strideY);
                float[,,] res = new float[resW, resH, Deep];

                //int dkw = (poolX - 1) / 2;
                //int dkh = (poolY - 1) / 2;

                int dkw = W % strideX;
                int dkh = H % strideY;

                for (int x = 0; x < resW; x++)
                    for (int y = 0; y < resH; y++)
                        for (int f = 0; f < Deep; f++)
                        {
                            float max = float.MinValue;                            
                            for (int kx = 0; kx < poolX; kx++)
                                for (int ky = 0; ky < poolY; ky++)
                                {
                                    int cX = x * strideX + kx - dkw;
                                    int cY = y * strideY + ky - dkh;
                                    float v = 0;
                                    if ((cX >= 0) && (cY >= 0) && (cX < W) && (cY < H))
                                    {
                                        v = src[cX, cY, f];
                                        if (v > max) max = v;
                                    }
                                }
                            res[x, y, f] = max;
                        }

                return res;
            }
        }

        /*
        protected float[,,] MaxPool2d(int poolX, int poolY, int strideX, int strideY, float[,,] src)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);            

            //int resW = W / strideX;
            //int resH = H / strideY;
            int resW = 1 + (W - poolX) / strideX;
            int resH = 1 + (H - poolX) / strideY;
            float[,,] res = new float[resW, resH, Deep];
            for (int d = 0; d < Deep; d++)
                for (int x = 0; x < resW; x++)
                    for (int y = 0; y < resH; y++)
                    {
                        float m = float.MinValue;
                        for (int i = x * strideX; i < x * strideX + poolX; i++)
                            for (int j = y * strideX; j < y * strideX + poolY; j++)
                                if (src[i, j, d] > m) m = src[i, j, d];
                        res[x, y, d] = m;
                    }
            return res;
        }*/

        protected float[] Dense1D(float[] src, float[,] weights, float[] bias)
        {
            int srcNmb = weights.GetLength(0);
            int dstNmb = weights.GetLength(1);

            float[] res = new float[dstNmb];

            for (int i = 0; i < dstNmb; i++)
            {
                res[i] = bias[i];
                for (int j = 0; j < srcNmb; j++)
                    res[i] += src[j] * weights[j, i];
            }

            return res;
        }

        protected float[,,] Conv2DTr(float[,,,] kernel, float[] bias, int strideX, int strideY, float[,,] src)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            // ядро свёртки 4-х мерное. X, Y, Канал, Кол-во фильтров
            int kernelW = kernel.GetLength(0);
            int kernelH = kernel.GetLength(1);
            int filterChannel = kernel.GetLength(2);
            int filterCount = kernel.GetLength(3);

            int resW = W * strideX;
            int resH = H * strideY;
            float[,,] res = new float[resW, resH, filterCount];

            for (int f = 0; f < filterCount; f++)
                for (int x = 0; x < resW; x++)
                    for (int y = 0; y < resH; y++)
                        res[x, y, f] = bias[f];


            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                    for (int f = 0; f < filterCount; f++)
                    {
                        for (int d = 0; d < Deep; d++)
                            for (int kx = 0; kx < kernelW; kx++)
                                for (int ky = 0; ky < kernelH; ky++)
                                    res[x * strideX + kx, y * strideY + ky, f] += src[x, y, d] * kernel[kx, ky, d, f];
                    }

            return res;
        }
    }
}
