
public class Complex {
	
	private double real, im;
	
	public Complex(double real, double im) {
		this.real = real;
		this.im = im;
	}
	
	public double getReal() {
		return real;
	}
	
	public double getImaginary() {
		return im;
	}
	
	public Complex square() {
		return new Complex(real*real - im*im, 2*real*im);
	}
	
	public double modulusSquared() {
		return real*real + im*im;
	}
	
	public Complex add(Complex d) {
		return new Complex(real + d.real, im + d.im);
	}
}
