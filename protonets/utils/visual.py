from visdom import Visdom
import numpy as np

viz = Visdom()

def train_val_acc(title):
    """

    """
    layout = dict(legend=['train', 'val'], title=title+" Acc Plot")
    return plotTwoLine(layout)

def train_val_loss(title):
    """
    
    """
    layout = dict(legend=['train', 'val'], title=title+" Loss Plot")
    return plotTwoLine(layout)

def plotTwoLine(layout):
    """

    """
    win = None
    prevX = 0
    prevYa = 0
    prevYb = 0
    
    def update_plotTwoLine(x, ya, yb):
        
        def update():
            nonlocal prevX, prevYa, prevYb, win
            if win is None:
                prevX = x
                prevYa = ya
                prevYb = yb
                win = viz.line(
                    X=np.column_stack((np.linspace(prevX, x, 10), np.linspace(prevX, x, 10))),
                    Y=np.column_stack((np.linspace(prevYa, ya, 10),
                        np.linspace(prevYb, yb, 10))),
                    opts=layout
                )
            else:
                viz.line(
                    X=np.column_stack((np.linspace(prevX, x, 10), np.linspace(prevX, x, 10))),
                    Y=np.column_stack((np.linspace(prevYa, ya, 10),
                            np.linspace(prevYb, yb, 10))),
                    win=win,
                    update='append'
                )    
        update()
        nonlocal prevX, prevYa, prevYb
        prevX = x
        prevYa = ya
        prevYb = yb
        
    return update_plotTwoLine

if __name__ == '__main__':
    from time import sleep
    pic = train_val_acc("First Plot")
    pic(1, 0.5, 0.1)
    sleep(1)
    pic(2, 0.2, 0.3)