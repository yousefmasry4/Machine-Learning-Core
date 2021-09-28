using System;

public class Wrist
{
    public double[] m1 = new double[2];
    public double[] m2 = new double[2];
    public double[] p1 = new double[2];
    public double[] p2 = new double[2];
    public double WristY;
    public double WristX;
    public double WristZ; 
    public double shapeY;
    public double shapeX;

    public Wrist(double[] m1, double[] m2, double WristY, double WristX,double WristZ, double shapeY, double shapeX, double[] p1, double[] p2)
    {
        this.m1 = m1;
        this.m2 = m2;
        this.WristY = WristY;
        this.WristX = WristX;
        this.WristZ = WristZ;
        this.shapeY = shapeY;
        this.shapeX = shapeX;
        this.p1 = p1;
        this.p2 = p2;
        
    }
    public bool CheckPalmorBack(double[] p1, double[] p2)
    {
        double d = Math.Pow(0 - 0, 2) + Math.Pow(p1[0] - p1[1], 2);
        double dis = Math.Sqrt(d);
        double d1 = Math.Pow(0 - 0, 2) + Math.Pow(p2[0] - p2[1], 2);
        double dis1 = Math.Sqrt(d);
        if (dis < dis1)
        {
            return true; 
        }
        return false; 
    }
    public double getlen(double[] x, double[] y)
    {
        double d = Math.Pow(x[0] - x[1], 2) + Math.Pow(y[0] - y[1], 2);
        double dis = Math.Sqrt(d);
        dis *= shapeY;
        dis /= 2;
        if (dis > 70)
        {
            dis = 70;
        }
        return dis;
    }

    public double[] lstsq(double[] x, double[] y)
    {
        double m = (y[0] - y[1]) / (x[0] - x[1]);
        double[] temp = new double[2] { m, y[0] - (x[0] * m) };
        return temp;
    }

    public double[] midpoint(double[] p1, double[] p2)
    {
        double[] temp = new double[2] { (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 };
        return temp;
    }
    public double[] WirstLandMarks()
    {
        double[] Mid = new double[2];
        Mid = midpoint(m1, m2);

        double[] x_coords = new double[2] { Mid[0] * shapeY, WristX * shapeY };
        double[] y_coords = new double[2] { Mid[1] * shapeX, WristY * shapeX };
        double[] A = new double[2] { WristX, WristY };
        double newx = 0;
        double newy = 0;
        double[] temp = new double[2];
        temp = lstsq(x_coords, y_coords);
        //Console.Write(temp[0]);
        //Console.WriteLine(" ");
        //Console.Write(temp[1]);
        if (WristX < Mid[0])
        {
            newx = WristX + (getlen(A, Mid) / shapeY);
        }
        else
        {
            newx = WristX - (getlen(A, Mid) / shapeY);
        }

        double newY = 0;
        double[] mc = new double[2];
        mc = lstsq(x_coords, y_coords);
        double m = mc[0];
        double c = mc[1];
        newY = m * newx + c;
        double a = WristX * shapeY;
        double b = WristY * shapeX;
        double r = getlen(A, Mid);
        double z = c - b;
        double ceq = 4 * ((Math.Pow(a, 2) + Math.Pow(z, 2)) - Math.Pow(r, 2)) * (1 + Math.Pow(m, 2));
        double beq = ((-2 * a) + (2 * z * m));
        //Console.WriteLine((1 + (m * m)));
        double x = (-1 * beq - Math.Sqrt((beq * beq) - ceq)) / (2 * (1 + (m * m)));
        double y = (x * m) + c;
        double[] wristpoints = new double[3] { x, y,WristZ*shapeX };
        //double ans = Math.Pow(r, 2) - Math.Pow(a, 2) - Math.Pow(c, 2) - Math.Pow(b, 2) + 2 * b * c;
        return wristpoints;
    }
    public double [] AngleOfRotation (double[] newCoordinates, double[] oldCoordinates))
        {
            double[] Angles = new double[3];
            Angle[0]=((newCoordinates[1]/oldCoordinates[2])+(newCoordinates[2]/oldCoordinates[1]))/((oldCoordinates[1]/oldCoordinates[2])+(oldCoordinates[1]/oldCoordinates[2]));
            Angle[1]=((newCoordinates[0]/oldCoordinates[2])+(newCoordinates[2]/oldCoordinates[0]))/((oldCoordinates[0]/oldCoordinates[2])+(oldCoordinates[1]/oldCoordinates[0]));
            Angle[2]=((newCoordinates[0]/oldCoordinates[1])+(newCoordinates[1]/oldCoordinates[0]))/((oldCoordinates[0]/oldCoordinates[1])+(oldCoordinates[1]/oldCoordinates[0]));
            return Angles; 
        } 
}