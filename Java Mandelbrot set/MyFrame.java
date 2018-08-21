import javax.swing.*;

import java.awt.*;
import java.awt.event.*;

class MyFrame extends JFrame {

	private double x, y;
	private int iter = 100;
	private double realMin = -2.0;
	private double realMax = 2.0;
	private double imMin = -1.6;
	private double imMax = 1.6;
	private JTextField xField, yField, iField, minRealField, maxRealField, minImField, maxImField;
	private MPanel mandelbrot;
	private JuliaPanel juliaPanel;
	private Complex d = new Complex(0, 0);
	
	public MyFrame(String title) {
		super(title);
    }

	public void init() {
		Container p = this.getContentPane();
		mandelbrot = new MPanel();
		juliaPanel = new JuliaPanel();
		p.setLayout(new GridLayout(1, 2));
		JPanel m = new JPanel(new BorderLayout());
		JPanel j = new JPanel(new BorderLayout());
		JPanel belowm = new JPanel();
		JPanel belowj = new JPanel();
		JPanel input = new JPanel(new GridLayout(2, 1));
		JPanel whiteSpace = new JPanel();
		
		JLabel xLabel = new JLabel("X:");
		JLabel yLabel = new JLabel(" Y:");
		JLabel iLabel = new JLabel(" Iterations:");
		JLabel minX = new JLabel("Min X:");
		JLabel maxX = new JLabel(" Max X:");
		JLabel minY = new JLabel(" Min Y:");
		JLabel maxY = new JLabel(" Max Y:");
		xField = new JTextField(12);
		yField = new JTextField(12);
		iField = new JTextField(5);
		minRealField = new JTextField(5);
		maxRealField = new JTextField(5);
		minImField = new JTextField(5);
		maxImField = new JTextField(5);
		JButton ok = new JButton("OK");
		JButton axisOk = new JButton("OK");
		xField.setEditable(false);
		yField.setEditable(false);
		
		input.add(belowm);
		input.add(belowj);
		belowm.add(xLabel);
		belowm.add(xField);
		belowm.add(yLabel);
		belowm.add(yField);
		belowm.add(iLabel);
		belowm.add(iField);
		belowj.add(minX);
		belowj.add(minRealField);
		belowj.add(maxX);
		belowj.add(maxRealField);
		belowj.add(minY);
		belowj.add(minImField);
		belowj.add(maxY);
		belowj.add(maxImField);
		belowm.add(ok);
		belowj.add(axisOk);
		
		ok.addActionListener( new ActionListener() {

		    public void actionPerformed(ActionEvent e){
		    	iter = Integer.parseInt(iField.getText());
		    	mandelbrot.repaint();
		    	juliaPanel.repaint();
		    }
		}
		);
		
		axisOk.addActionListener( new ActionListener() {

			public void actionPerformed(ActionEvent e) {
				realMin = Double.parseDouble((minRealField.getText()));
				realMax = Double.parseDouble((maxRealField.getText()));
				imMin = Double.parseDouble((minImField.getText()));
				imMax = Double.parseDouble((maxImField.getText()));
				mandelbrot.repaint();
			}
		}
		);
		
		mandelbrot.addMouseListener( new MouseListener() {

			public void mouseClicked(MouseEvent e) {
				double realAxis =  (e.getX()*(realMax - realMin))/mandelbrot.getWidth() + realMin;
				double imAxis = -(e.getY()*(imMax - imMin))/mandelbrot.getHeight() + imMax;
				String sx = Double.toString(realAxis);
				String sy = Double.toString(imAxis);
				xField.setText(sx);
				yField.setText(sy);
				d = new Complex(realAxis, imAxis);
				juliaPanel.repaint();
			}
			
			public void mouseEntered(MouseEvent e) {}
			public void mouseExited(MouseEvent e) {}
			public void mousePressed(MouseEvent e) {
				x =  (e.getX()*(realMax - realMin))/mandelbrot.getWidth() + realMin;
				y = -(e.getY()*(imMax - imMin))/mandelbrot.getHeight() + imMax;
			}
			
			public void mouseReleased(MouseEvent e) {
				double realAxis =  (e.getX()*(realMax - realMin))/mandelbrot.getWidth() + realMin;
				double imAxis = -(e.getY()*(imMax - imMin))/mandelbrot.getHeight() + imMax;
				if (realAxis > x) {
					if (imAxis > y) {
						realMin = x;
						imMin = y;
						realMax = realAxis;
						imMax = imAxis;
					} else if (imAxis < y) {
						realMin = x;
						imMin = imAxis;
						realMax = realAxis;
						imMax = y;
					}
				} else if (realAxis < x) {
					if (imAxis > y) {
						realMin = realAxis;
						imMin = y;
						realMax = x;
						imMax = imAxis;
					} else if (imAxis < y) {
						realMin = realAxis;
						imMin = imAxis;
						realMax = x;
						imMax = y;
					}
				}
				mandelbrot.repaint();
			}
			}
		);
		
		p.add(m);
		p.add(j);
		m.add(mandelbrot, BorderLayout.CENTER);
		m.add(belowm, BorderLayout.SOUTH);
		m.add(whiteSpace, BorderLayout.EAST);
		j.add(whiteSpace, BorderLayout.WEST);
		j.add(juliaPanel, BorderLayout.CENTER);
		j.add(belowj, BorderLayout.SOUTH);
		
		this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE); 	
		this.setSize(1030, 500);
		//this.pack();
		this.setVisible(true);
    }
	
	abstract class FractalPanel extends JPanel {
		
		Complex c;
		
		public FractalPanel() {
			setPreferredSize(new Dimension(100, 300));
		}
		
		public void paintComponent(Graphics g) {
			super.paintComponent(g);
			for (int i = 0; i < getWidth(); i++) {
				for (int j = 0; j < getHeight(); j++) {
					g.setColor(getPixelColour(i, j));
					g.fillRect(i, j, 1, 1);
				}
			}
		}
		
		public Color getPixelColour(int x, int y) {
			double realX = (x*(getRealMax() - getRealMin()))/getWidth() + getRealMin();
			double imY = -(y*(getImMax() - getImMin()))/getHeight() + getImMax();
			Complex pixel = new Complex(realX, imY);
			c = pixel;
			
			if (optimiseMandelbrot(realX, imY) == 1) {
				return new Color(0, 0, 0);
			}
			
			int i = 0;
			for (; i < iter; i++) {
				if (pixel.modulusSquared() > 4) {
					break;
				}
				pixel = fractalFormula(pixel);
			}
			
			if (i == iter) {
				return new Color(0, 0, 0);
			}
			
			double aa = i + 1 - Math.log(Math.log(Math.sqrt(pixel.modulusSquared())))/Math.log(2);
			float colorGradient;
			if (iter > 100) {
				colorGradient = 100;
			} else {
				colorGradient = iter;
			}
			return Color.getHSBColor((float)aa/colorGradient, 1.0F, 1.0F);
		}
		
		abstract public Complex fractalFormula(Complex f);
		abstract public int optimiseMandelbrot(double realX, double imY);
		abstract public double getRealMax();
		abstract public double getRealMin();
		abstract public double getImMax();
		abstract public double getImMin();
	}
		
	abstract class ZoomableFractal extends FractalPanel {
		public double getRealMax() {
			return realMax;
		}

		public double getRealMin() {
			return realMin;
		}

		public double getImMax() {
			return imMax;
		}

		public double getImMin() {
			return imMin;
		}
	}
	
	abstract class UnzoomableFractal extends FractalPanel {
		
		private final double realMax = 2.0;
		private final double realMin = -2.0;
		private final double imMax = 1.6;
		private final double imMin = -1.6;
		
		public double getRealMax() {
			return realMax;
		}

		public double getRealMin() {
			return realMin;
		}

		public double getImMax() {
			return imMax;
		}

		public double getImMin() {
			return imMin;
		}
	}
	
	class MPanel extends ZoomableFractal {
		
		public Complex fractalFormula(Complex f) {
			return f.square().add(c);
		}
		
		public int optimiseMandelbrot(double realX, double imY) {
			double p = (realX - 0.25)*(realX - 0.25) + imY*imY;
			if (p*(p + (realX - 0.25)) < 0.25*imY*imY || (realX + 1)*(realX + 1) + imY*imY < 0.0625) {
				return 1;
			} else {
				return 0;
			}
		}
	}
	
	class JuliaPanel extends UnzoomableFractal {
		
		public Complex fractalFormula(Complex f) {
			return f.square().add(d);
		}

		public int optimiseMandelbrot(double realX, double imY) {return 0;}
	}
}