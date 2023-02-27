using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace _2
{
    class Program
    {
        static void Main()
        {
            // Загрузка классификатора «HaarCascadeClassifier» для распознавания лиц
            var faceCascadeClassifier = new CascadeClassifier("haarcascade_frontalface_default.xml");

            // Загрузка изображения, на котором необходимо распознать лицо
            var image = new Image<Bgr, byte>("image.jpg");
            // Преобразование изображения в полутоновое
            var grayImage = image.Convert<Gray, byte>();

            // Распознавание лиц на полутоновом изображении
            var faces = faceCascadeClassifier.DetectMultiScale(grayImage, 1.1, 5);

            // Прорисовка прямоугольника вокруг найденного лица
            foreach (var face in faces)
            {
                image.Draw(face, new Bgr(Color.Red), 2);
            }

            // Отображение результата
            CvInvoke.Imshow("Face detection", image);
            CvInvoke.WaitKey();
        }
    }

}
