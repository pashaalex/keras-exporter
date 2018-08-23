using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;

namespace MyModel
{
    public static class NetUtils
    {
        public static int MaxIndex1D(float[] src)
        {
            int ind = 0;
            for (int i = 0; i < src.Length; i++)
                if (src[ind] < src[i])
                    ind = i;
            return ind;
        }

        public static void Write3D_AS_2D(float[,,] src, int dim, string fName, int maxDim = int.MaxValue)
        {
            int W = Math.Min(maxDim, src.GetLength(0));
            int H = Math.Min(maxDim, src.GetLength(1));

            using (StreamWriter sw = new StreamWriter(fName))
            {
                for (int x = 0; x < W; x++)
                {
                    for (int y = 0; y < H; y++)
                        sw.Write((src[x, y, dim].ToString("0.00") + " "));
                    sw.WriteLine();                    
                }
            }
        }

        public static unsafe float[,,] PrepareImageGray(string fName)
        {
            using (Bitmap bmp = (Bitmap)Bitmap.FromFile(fName))
                return PrepareImageGray(bmp);
        }

        public static unsafe float[,,] PrepareImageRGB(string fName)
        {
            using (Bitmap bmp = (Bitmap)Bitmap.FromFile(fName))
                return PrepareImageRGB(bmp);
        }


        public static unsafe float[,,] PrepareImageGray(Bitmap bmp)
        {
            int W = bmp.Width;
            int H = bmp.Height;
            float[,,] src = new float[H, W, 1];
            BitmapData bData = bmp.LockBits(new Rectangle(0, 0, W, H), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            for (int Y = 0; Y < H; Y++)
            {
                byte* row_bytes = (byte*)bData.Scan0.ToPointer() + Y * bData.Stride;
                for (int X = 0; X < W; X++)
                    src[Y, X, 0] = row_bytes[X * 3] / 255.0F;
            }
            bmp.UnlockBits(bData);
            return src;
        }

        public static unsafe float[,,] PrepareImageRGB(Bitmap bmp)
        {
            int W = bmp.Width;
            int H = bmp.Height;
            float[,,] src = new float[W, H, 3];
            BitmapData bData = bmp.LockBits(new Rectangle(0, 0, W, H), ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            for (int Y = 0; Y < H; Y++)
            {
                byte* row_bytes = (byte*)bData.Scan0.ToPointer() + Y * bData.Stride;
                for (int X = 0; X < W; X++)
                {
                    src[Y, X, 0] = row_bytes[X * 3 + 0];
                    src[Y, X, 1] = row_bytes[X * 3 + 1];
                    src[Y, X, 2] = row_bytes[X * 3 + 2];
                }
            }
            bmp.UnlockBits(bData);
            return src;
        }


        public static float[,,] Concatenate3D(float[,,] a, float[,,] b)
        {
            if (a.GetLength(0) != b.GetLength(0)) throw new Exception("Dimension mismatch");
            if (a.GetLength(1) != b.GetLength(1)) throw new Exception("Dimension mismatch");

            int W = a.GetLength(0);
            int H = a.GetLength(1);
            int Da = a.GetLength(2);

            int Db = b.GetLength(2);
            float[,,] res = new float[W, H, Da + Db];
            for (int d = 0; d < Da; d++)
                for (int x = 0; x < W; x++)
                    for (int y = 0; y < H; y++)
                        res[x, y, d] = a[x, y, d];

            for (int d = 0; d < Db; d++)
                for (int x = 0; x < W; x++)
                    for (int y = 0; y < H; y++)
                        res[x, y, d + Da] = b[x, y, d];

            return res;
        }

        public static Array ReadTensor(string fName)
        {
            using (FileStream fs = new FileStream(fName, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader(fs))
                return ReadTensor(br);
        }

        public static Array ReadTensor(BinaryReader br)
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

        public static float[,,] ReadTensor3D(string fName)
        {
            using (FileStream fs = new FileStream(fName, FileMode.Open, FileAccess.Read))
            using (BinaryReader br = new BinaryReader(fs))
            {
                int d = br.ReadInt32();
                if (d != 3) throw new Exception("Dimension mismatch");
                int W = br.ReadInt32();
                int H = br.ReadInt32();
                int Deep = br.ReadInt32();

                float[,,] res = new float[W, H, Deep];

                for (int x = 0; x < W; x++)
                    for (int y = 0; y < H; y++)
                        for (int dp = 0; dp < Deep; dp++)
                            res[x, y, dp] = br.ReadSingle();
                return res;                                
            }
        }


        public static void WriteTensor3D(float[,,] src, string fName, bool append)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);
            FileStream fs = null;
            try
            {
                if (append)
                    fs = new FileStream(fName, FileMode.Append);
                else
                    fs = new FileStream(fName, FileMode.Create);

                using (BinaryWriter bw = new BinaryWriter(fs))
                    for (int x = 0; x < W; x++)
                        for (int y = 0; y < H; y++)
                            for (int d = 0; d < Deep; d++)
                                bw.Write(src[x, y, d]);
            }
            finally
            {
                fs.Dispose();
            }
        }


        public static double CompareTensor1DAvgDiff(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new Exception("Dimension mismatch");

            double sum = 0;
            double cnt = 0;
            for (int x = 0; x < a.Length; x++)
            {
                sum = sum + Math.Abs(a[x] - b[x]);
                cnt += 1;
            }
            return sum / cnt;
        }

        public static double CompareTensor1DMaxDiff(float[] a, float[] b)
        {
            if (a.Length != b.Length) throw new Exception("Dimension mismatch");

            double max = 0;            
            for (int x = 0; x < a.Length; x++)
            {
                float f = Math.Abs(a[x] - b[x]);
                if (f > max) max = f;
            }
            return max;
        }



        public static double CompareTensor3DAvgDiff(float[,,] a, float[,,] b)
        {
            if (a.GetLength(0) != b.GetLength(0)) throw new Exception("Dimension mismatch");
            if (a.GetLength(1) != b.GetLength(1)) throw new Exception("Dimension mismatch");
            if (a.GetLength(2) != b.GetLength(2)) throw new Exception("Dimension mismatch");

            int W = a.GetLength(0);
            int H = a.GetLength(1);
            int Deep = a.GetLength(2);
            double sum = 0;
            double cnt = 0;
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                    for (int d = 0; d < Deep; d++)
                    {
                        sum = sum + Math.Abs(a[x, y, d] - b[x, y, d]);
                        cnt += 1;
                    }
            return sum / cnt;
        }

        public static float CompareTensor3DMaxDiff(float[,,] a, float[,,] b)
        {
            if (a.GetLength(0) != b.GetLength(0)) throw new Exception("Dimension mismatch");
            if (a.GetLength(1) != b.GetLength(1)) throw new Exception("Dimension mismatch");
            if (a.GetLength(2) != b.GetLength(2)) throw new Exception("Dimension mismatch");

            int W = a.GetLength(0);
            int H = a.GetLength(1);
            int Deep = a.GetLength(2);
            float max = 0;
            
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                    for (int d = 0; d < Deep; d++)
                    {
                        float f = Math.Abs(a[x, y, d] - b[x, y, d]);
                        if (f > max) max = f;
                    }
            return max;
        }


        public static bool CompareTensor3D(float[,,] a, float[,,] b, float threshold)
        {
            if (a.GetLength(0) != b.GetLength(0)) throw new Exception("Dimension mismatch");
            if (a.GetLength(1) != b.GetLength(1)) throw new Exception("Dimension mismatch");
            if (a.GetLength(2) != b.GetLength(2)) throw new Exception("Dimension mismatch");

            int W = a.GetLength(0);
            int H = a.GetLength(1);
            int Deep = a.GetLength(2);
            for (int x = 0; x < W; x++)
                for (int y = 0; y < H; y++)
                    for (int d = 0; d < Deep; d++)
                        if (Math.Abs(a[x, y, d] - b[x, y, d]) > threshold)
                        {
                            Console.WriteLine("Avg=" + CompareTensor3DAvgDiff(a, b).ToString() + " Max=" + CompareTensor3DMaxDiff(a, b).ToString());
                            return false;
                        }
            return true;
        }

        public static unsafe Bitmap VisualizeOutput(float[,,] src)
        {
            int W = src.GetLength(0);
            int H = src.GetLength(1);
            int Deep = src.GetLength(2);

            int pad = 10;

            int CntWidth = (int)Math.Ceiling(Math.Sqrt(Deep)); // Image count
            int CntWidthD = (int)Math.Floor(Math.Sqrt(Deep));
            Bitmap bmp = new Bitmap((CntWidth) * (W + pad) + 4, CntWidth * (H + pad) + 4, PixelFormat.Format24bppRgb);
            BitmapData bData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadWrite, PixelFormat.Format24bppRgb);

            int Cx = 0;
            int Cy = 0;
            for (int FilterId = 0; FilterId < Deep; FilterId++)
            {
                float min = float.MaxValue;
                float max = float.MinValue;
                for (int y = 0; y < H; y++)
                    for (int x = 0; x < W; x++)
                    {
                        if (src[x, y, FilterId] < min) min = src[x, y, FilterId];
                        if (src[x, y, FilterId] > max) max = src[x, y, FilterId];
                    }


                for (int y = 0; y < H; y++)
                {
                    byte* row_bytes = (byte*)bData.Scan0.ToPointer() + (Cy * (H + pad) + y + pad) * bData.Stride + (Cx * (W + pad) + pad) * 3;
                    for (int x = 0; x < W; x++)
                    {
                        byte b = (byte)(255.0F * (src[x, y, FilterId] - min) / (max - min));
                        row_bytes[x * 3 + 0] = b;
                        row_bytes[x * 3 + 1] = b;
                        row_bytes[x * 3 + 2] = b;
                    }
                }

                Cx = Cx + 1;
                if ((Cx > CntWidthD) || ((Cx == CntWidthD) && (CntWidthD == CntWidth)))
                {
                    Cx = 0;
                    Cy = Cy + 1;
                }
            }
            bmp.UnlockBits(bData);

            // Draw borders
            using (Graphics g = Graphics.FromImage(bmp))
            {
                Cx = 0;
                Cy = 0;
                Font f = new Font(FontFamily.GenericSerif, 8);
                for (int FilterId = 0; FilterId < Deep; FilterId++)
                {
                    g.DrawRectangle(Pens.White, Cx * (W + pad) + pad, Cy * (H + pad) + pad, W, H);
                    g.DrawString(FilterId.ToString(), f, Brushes.White, Cx * (W + pad), Cy * (H + pad));
                    Cx = Cx + 1;
                    if ((Cx > CntWidthD) || ((Cx == CntWidthD) && (CntWidthD == CntWidth)))
                    {
                        Cx = 0;
                        Cy = Cy + 1;
                    }
                }
            }

            return bmp;
        }

        public static string[] DecodeImageNetResult(float[] predictions, int maxElements)
        {
            return predictions
                .Zip(ImageNet_Classes, (a, b) => new { Score = a, Value = b })
                .OrderByDescending(n => n.Score)
                .Take(maxElements)
                .Select(n => n.Value + " [" + n.Score.ToString() + "]")
                .ToArray();
        }

        public static string[] ImageNet_Classes = {
            "tench",
"goldfish",
"great_white_shark",
"tiger_shark",
"hammerhead",
"electric_ray",
"stingray",
"cock",
"hen",
"ostrich",
"brambling",
"goldfinch",
"house_finch",
"junco",
"indigo_bunting",
"robin",
"bulbul",
"jay",
"magpie",
"chickadee",
"water_ouzel",
"kite",
"bald_eagle",
"vulture",
"great_grey_owl",
"European_fire_salamander",
"common_newt",
"eft",
"spotted_salamander",
"axolotl",
"bullfrog",
"tree_frog",
"tailed_frog",
"loggerhead",
"leatherback_turtle",
"mud_turtle",
"terrapin",
"box_turtle",
"banded_gecko",
"common_iguana",
"American_chameleon",
"whiptail",
"agama",
"frilled_lizard",
"alligator_lizard",
"Gila_monster",
"green_lizard",
"African_chameleon",
"Komodo_dragon",
"African_crocodile",
"American_alligator",
"triceratops",
"thunder_snake",
"ringneck_snake",
"hognose_snake",
"green_snake",
"king_snake",
"garter_snake",
"water_snake",
"vine_snake",
"night_snake",
"boa_constrictor",
"rock_python",
"Indian_cobra",
"green_mamba",
"sea_snake",
"horned_viper",
"diamondback",
"sidewinder",
"trilobite",
"harvestman",
"scorpion",
"black_and_gold_garden_spider",
"barn_spider",
"garden_spider",
"black_widow",
"tarantula",
"wolf_spider",
"tick",
"centipede",
"black_grouse",
"ptarmigan",
"ruffed_grouse",
"prairie_chicken",
"peacock",
"quail",
"partridge",
"African_grey",
"macaw",
"sulphur-crested_cockatoo",
"lorikeet",
"coucal",
"bee_eater",
"hornbill",
"hummingbird",
"jacamar",
"toucan",
"drake",
"red-breasted_merganser",
"goose",
"black_swan",
"tusker",
"echidna",
"platypus",
"wallaby",
"koala",
"wombat",
"jellyfish",
"sea_anemone",
"brain_coral",
"flatworm",
"nematode",
"conch",
"snail",
"slug",
"sea_slug",
"chiton",
"chambered_nautilus",
"Dungeness_crab",
"rock_crab",
"fiddler_crab",
"king_crab",
"American_lobster",
"spiny_lobster",
"crayfish",
"hermit_crab",
"isopod",
"white_stork",
"black_stork",
"spoonbill",
"flamingo",
"little_blue_heron",
"American_egret",
"bittern",
"crane",
"limpkin",
"European_gallinule",
"American_coot",
"bustard",
"ruddy_turnstone",
"red-backed_sandpiper",
"redshank",
"dowitcher",
"oystercatcher",
"pelican",
"king_penguin",
"albatross",
"grey_whale",
"killer_whale",
"dugong",
"sea_lion",
"Chihuahua",
"Japanese_spaniel",
"Maltese_dog",
"Pekinese",
"Shih-Tzu",
"Blenheim_spaniel",
"papillon",
"toy_terrier",
"Rhodesian_ridgeback",
"Afghan_hound",
"basset",
"beagle",
"bloodhound",
"bluetick",
"black-and-tan_coonhound",
"Walker_hound",
"English_foxhound",
"redbone",
"borzoi",
"Irish_wolfhound",
"Italian_greyhound",
"whippet",
"Ibizan_hound",
"Norwegian_elkhound",
"otterhound",
"Saluki",
"Scottish_deerhound",
"Weimaraner",
"Staffordshire_bullterrier",
"American_Staffordshire_terrier",
"Bedlington_terrier",
"Border_terrier",
"Kerry_blue_terrier",
"Irish_terrier",
"Norfolk_terrier",
"Norwich_terrier",
"Yorkshire_terrier",
"wire-haired_fox_terrier",
"Lakeland_terrier",
"Sealyham_terrier",
"Airedale",
"cairn",
"Australian_terrier",
"Dandie_Dinmont",
"Boston_bull",
"miniature_schnauzer",
"giant_schnauzer",
"standard_schnauzer",
"Scotch_terrier",
"Tibetan_terrier",
"silky_terrier",
"soft-coated_wheaten_terrier",
"West_Highland_white_terrier",
"Lhasa",
"flat-coated_retriever",
"curly-coated_retriever",
"golden_retriever",
"Labrador_retriever",
"Chesapeake_Bay_retriever",
"German_short-haired_pointer",
"vizsla",
"English_setter",
"Irish_setter",
"Gordon_setter",
"Brittany_spaniel",
"clumber",
"English_springer",
"Welsh_springer_spaniel",
"cocker_spaniel",
"Sussex_spaniel",
"Irish_water_spaniel",
"kuvasz",
"schipperke",
"groenendael",
"malinois",
"briard",
"kelpie",
"komondor",
"Old_English_sheepdog",
"Shetland_sheepdog",
"collie",
"Border_collie",
"Bouvier_des_Flandres",
"Rottweiler",
"German_shepherd",
"Doberman",
"miniature_pinscher",
"Greater_Swiss_Mountain_dog",
"Bernese_mountain_dog",
"Appenzeller",
"EntleBucher",
"boxer",
"bull_mastiff",
"Tibetan_mastiff",
"French_bulldog",
"Great_Dane",
"Saint_Bernard",
"Eskimo_dog",
"malamute",
"Siberian_husky",
"dalmatian",
"affenpinscher",
"basenji",
"pug",
"Leonberg",
"Newfoundland",
"Great_Pyrenees",
"Samoyed",
"Pomeranian",
"chow",
"keeshond",
"Brabancon_griffon",
"Pembroke",
"Cardigan",
"toy_poodle",
"miniature_poodle",
"standard_poodle",
"Mexican_hairless",
"timber_wolf",
"white_wolf",
"red_wolf",
"coyote",
"dingo",
"dhole",
"African_hunting_dog",
"hyena",
"red_fox",
"kit_fox",
"Arctic_fox",
"grey_fox",
"tabby",
"tiger_cat",
"Persian_cat",
"Siamese_cat",
"Egyptian_cat",
"cougar",
"lynx",
"leopard",
"snow_leopard",
"jaguar",
"lion",
"tiger",
"cheetah",
"brown_bear",
"American_black_bear",
"ice_bear",
"sloth_bear",
"mongoose",
"meerkat",
"tiger_beetle",
"ladybug",
"ground_beetle",
"long-horned_beetle",
"leaf_beetle",
"dung_beetle",
"rhinoceros_beetle",
"weevil",
"fly",
"bee",
"ant",
"grasshopper",
"cricket",
"walking_stick",
"cockroach",
"mantis",
"cicada",
"leafhopper",
"lacewing",
"dragonfly",
"damselfly",
"admiral",
"ringlet",
"monarch",
"cabbage_butterfly",
"sulphur_butterfly",
"lycaenid",
"starfish",
"sea_urchin",
"sea_cucumber",
"wood_rabbit",
"hare",
"Angora",
"hamster",
"porcupine",
"fox_squirrel",
"marmot",
"beaver",
"guinea_pig",
"sorrel",
"zebra",
"hog",
"wild_boar",
"warthog",
"hippopotamus",
"ox",
"water_buffalo",
"bison",
"ram",
"bighorn",
"ibex",
"hartebeest",
"impala",
"gazelle",
"Arabian_camel",
"llama",
"weasel",
"mink",
"polecat",
"black-footed_ferret",
"otter",
"skunk",
"badger",
"armadillo",
"three-toed_sloth",
"orangutan",
"gorilla",
"chimpanzee",
"gibbon",
"siamang",
"guenon",
"patas",
"baboon",
"macaque",
"langur",
"colobus",
"proboscis_monkey",
"marmoset",
"capuchin",
"howler_monkey",
"titi",
"spider_monkey",
"squirrel_monkey",
"Madagascar_cat",
"indri",
"Indian_elephant",
"African_elephant",
"lesser_panda",
"giant_panda",
"barracouta",
"eel",
"coho",
"rock_beauty",
"anemone_fish",
"sturgeon",
"gar",
"lionfish",
"puffer",
"abacus",
"abaya",
"academic_gown",
"accordion",
"acoustic_guitar",
"aircraft_carrier",
"airliner",
"airship",
"altar",
"ambulance",
"amphibian",
"analog_clock",
"apiary",
"apron",
"ashcan",
"assault_rifle",
"backpack",
"bakery",
"balance_beam",
"balloon",
"ballpoint",
"Band_Aid",
"banjo",
"bannister",
"barbell",
"barber_chair",
"barbershop",
"barn",
"barometer",
"barrel",
"barrow",
"baseball",
"basketball",
"bassinet",
"bassoon",
"bathing_cap",
"bath_towel",
"bathtub",
"beach_wagon",
"beacon",
"beaker",
"bearskin",
"beer_bottle",
"beer_glass",
"bell_cote",
"bib",
"bicycle-built-for-two",
"bikini",
"binder",
"binoculars",
"birdhouse",
"boathouse",
"bobsled",
"bolo_tie",
"bonnet",
"bookcase",
"bookshop",
"bottlecap",
"bow",
"bow_tie",
"brass",
"brassiere",
"breakwater",
"breastplate",
"broom",
"bucket",
"buckle",
"bulletproof_vest",
"bullet_train",
"butcher_shop",
"cab",
"caldron",
"candle",
"cannon",
"canoe",
"can_opener",
"cardigan",
"car_mirror",
"carousel",
"carpenter's_kit",
"carton",
"car_wheel",
"cash_machine",
"cassette",
"cassette_player",
"castle",
"catamaran",
"CD_player",
"cello",
"cellular_telephone",
"chain",
"chainlink_fence",
"chain_mail",
"chain_saw",
"chest",
"chiffonier",
"chime",
"china_cabinet",
"Christmas_stocking",
"church",
"cinema",
"cleaver",
"cliff_dwelling",
"cloak",
"clog",
"cocktail_shaker",
"coffee_mug",
"coffeepot",
"coil",
"combination_lock",
"computer_keyboard",
"confectionery",
"container_ship",
"convertible",
"corkscrew",
"cornet",
"cowboy_boot",
"cowboy_hat",
"cradle",
"crane",
"crash_helmet",
"crate",
"crib",
"Crock_Pot",
"croquet_ball",
"crutch",
"cuirass",
"dam",
"desk",
"desktop_computer",
"dial_telephone",
"diaper",
"digital_clock",
"digital_watch",
"dining_table",
"dishrag",
"dishwasher",
"disk_brake",
"dock",
"dogsled",
"dome",
"doormat",
"drilling_platform",
"drum",
"drumstick",
"dumbbell",
"Dutch_oven",
"electric_fan",
"electric_guitar",
"electric_locomotive",
"entertainment_center",
"envelope",
"espresso_maker",
"face_powder",
"feather_boa",
"file",
"fireboat",
"fire_engine",
"fire_screen",
"flagpole",
"flute",
"folding_chair",
"football_helmet",
"forklift",
"fountain",
"fountain_pen",
"four-poster",
"freight_car",
"French_horn",
"frying_pan",
"fur_coat",
"garbage_truck",
"gasmask",
"gas_pump",
"goblet",
"go-kart",
"golf_ball",
"golfcart",
"gondola",
"gong",
"gown",
"grand_piano",
"greenhouse",
"grille",
"grocery_store",
"guillotine",
"hair_slide",
"hair_spray",
"half_track",
"hammer",
"hamper",
"hand_blower",
"hand-held_computer",
"handkerchief",
"hard_disc",
"harmonica",
"harp",
"harvester",
"hatchet",
"holster",
"home_theater",
"honeycomb",
"hook",
"hoopskirt",
"horizontal_bar",
"horse_cart",
"hourglass",
"iPod",
"iron",
"jack-o'-lantern",
"jean",
"jeep",
"jersey",
"jigsaw_puzzle",
"jinrikisha",
"joystick",
"kimono",
"knee_pad",
"knot",
"lab_coat",
"ladle",
"lampshade",
"laptop",
"lawn_mower",
"lens_cap",
"letter_opener",
"library",
"lifeboat",
"lighter",
"limousine",
"liner",
"lipstick",
"Loafer",
"lotion",
"loudspeaker",
"loupe",
"lumbermill",
"magnetic_compass",
"mailbag",
"mailbox",
"maillot",
"maillot",
"manhole_cover",
"maraca",
"marimba",
"mask",
"matchstick",
"maypole",
"maze",
"measuring_cup",
"medicine_chest",
"megalith",
"microphone",
"microwave",
"military_uniform",
"milk_can",
"minibus",
"miniskirt",
"minivan",
"missile",
"mitten",
"mixing_bowl",
"mobile_home",
"Model_T",
"modem",
"monastery",
"monitor",
"moped",
"mortar",
"mortarboard",
"mosque",
"mosquito_net",
"motor_scooter",
"mountain_bike",
"mountain_tent",
"mouse",
"mousetrap",
"moving_van",
"muzzle",
"nail",
"neck_brace",
"necklace",
"nipple",
"notebook",
"obelisk",
"oboe",
"ocarina",
"odometer",
"oil_filter",
"organ",
"oscilloscope",
"overskirt",
"oxcart",
"oxygen_mask",
"packet",
"paddle",
"paddlewheel",
"padlock",
"paintbrush",
"pajama",
"palace",
"panpipe",
"paper_towel",
"parachute",
"parallel_bars",
"park_bench",
"parking_meter",
"passenger_car",
"patio",
"pay-phone",
"pedestal",
"pencil_box",
"pencil_sharpener",
"perfume",
"Petri_dish",
"photocopier",
"pick",
"pickelhaube",
"picket_fence",
"pickup",
"pier",
"piggy_bank",
"pill_bottle",
"pillow",
"ping-pong_ball",
"pinwheel",
"pirate",
"pitcher",
"plane",
"planetarium",
"plastic_bag",
"plate_rack",
"plow",
"plunger",
"Polaroid_camera",
"pole",
"police_van",
"poncho",
"pool_table",
"pop_bottle",
"pot",
"potter's_wheel",
"power_drill",
"prayer_rug",
"printer",
"prison",
"projectile",
"projector",
"puck",
"punching_bag",
"purse",
"quill",
"quilt",
"racer",
"racket",
"radiator",
"radio",
"radio_telescope",
"rain_barrel",
"recreational_vehicle",
"reel",
"reflex_camera",
"refrigerator",
"remote_control",
"restaurant",
"revolver",
"rifle",
"rocking_chair",
"rotisserie",
"rubber_eraser",
"rugby_ball",
"rule",
"running_shoe",
"safe",
"safety_pin",
"saltshaker",
"sandal",
"sarong",
"sax",
"scabbard",
"scale",
"school_bus",
"schooner",
"scoreboard",
"screen",
"screw",
"screwdriver",
"seat_belt",
"sewing_machine",
"shield",
"shoe_shop",
"shoji",
"shopping_basket",
"shopping_cart",
"shovel",
"shower_cap",
"shower_curtain",
"ski",
"ski_mask",
"sleeping_bag",
"slide_rule",
"sliding_door",
"slot",
"snorkel",
"snowmobile",
"snowplow",
"soap_dispenser",
"soccer_ball",
"sock",
"solar_dish",
"sombrero",
"soup_bowl",
"space_bar",
"space_heater",
"space_shuttle",
"spatula",
"speedboat",
"spider_web",
"spindle",
"sports_car",
"spotlight",
"stage",
"steam_locomotive",
"steel_arch_bridge",
"steel_drum",
"stethoscope",
"stole",
"stone_wall",
"stopwatch",
"stove",
"strainer",
"streetcar",
"stretcher",
"studio_couch",
"stupa",
"submarine",
"suit",
"sundial",
"sunglass",
"sunglasses",
"sunscreen",
"suspension_bridge",
"swab",
"sweatshirt",
"swimming_trunks",
"swing",
"switch",
"syringe",
"table_lamp",
"tank",
"tape_player",
"teapot",
"teddy",
"television",
"tennis_ball",
"thatch",
"theater_curtain",
"thimble",
"thresher",
"throne",
"tile_roof",
"toaster",
"tobacco_shop",
"toilet_seat",
"torch",
"totem_pole",
"tow_truck",
"toyshop",
"tractor",
"trailer_truck",
"tray",
"trench_coat",
"tricycle",
"trimaran",
"tripod",
"triumphal_arch",
"trolleybus",
"trombone",
"tub",
"turnstile",
"typewriter_keyboard",
"umbrella",
"unicycle",
"upright",
"vacuum",
"vase",
"vault",
"velvet",
"vending_machine",
"vestment",
"viaduct",
"violin",
"volleyball",
"waffle_iron",
"wall_clock",
"wallet",
"wardrobe",
"warplane",
"washbasin",
"washer",
"water_bottle",
"water_jug",
"water_tower",
"whiskey_jug",
"whistle",
"wig",
"window_screen",
"window_shade",
"Windsor_tie",
"wine_bottle",
"wing",
"wok",
"wooden_spoon",
"wool",
"worm_fence",
"wreck",
"yawl",
"yurt",
"web_site",
"comic_book",
"crossword_puzzle",
"street_sign",
"traffic_light",
"book_jacket",
"menu",
"plate",
"guacamole",
"consomme",
"hot_pot",
"trifle",
"ice_cream",
"ice_lolly",
"French_loaf",
"bagel",
"pretzel",
"cheeseburger",
"hotdog",
"mashed_potato",
"head_cabbage",
"broccoli",
"cauliflower",
"zucchini",
"spaghetti_squash",
"acorn_squash",
"butternut_squash",
"cucumber",
"artichoke",
"bell_pepper",
"cardoon",
"mushroom",
"Granny_Smith",
"strawberry",
"orange",
"lemon",
"fig",
"pineapple",
"banana",
"jackfruit",
"custard_apple",
"pomegranate",
"hay",
"carbonara",
"chocolate_sauce",
"dough",
"meat_loaf",
"pizza",
"potpie",
"burrito",
"red_wine",
"espresso",
"cup",
"eggnog",
"alp",
"bubble",
"cliff",
"coral_reef",
"geyser",
"lakeside",
"promontory",
"sandbar",
"seashore",
"valley",
"volcano",
"ballplayer",
"groom",
"scuba_diver",
"rapeseed",
"daisy",
"yellow_lady's_slipper",
"corn",
"acorn",
"hip",
"buckeye",
"coral_fungus",
"agaric",
"gyromitra",
"stinkhorn",
"earthstar",
"hen-of-the-woods",
"bolete",
"ear",
"toilet_tissue"
        };


    }
}
