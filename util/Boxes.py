# Contains classes for bounding boxes, as well as methods for calculating the area of intersection and union
# Target boxes can be assigned another bounding box - useful for calculation IoU scores

class BoundingBox():
    def __init__(self, x, y, width, height):
        #store x and y of top left corner, as well as width and height of box
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def intersect(self, box2, recursive = False):
        #Returns the area of the intersection of two boxes

        #[top left, top right, bottom left, bottom right] 
        b1Corners = [(self.x, self.y), (self.x+self.width, self.y), (self.x, self.y+self.height), (self.x+self.width, self.y+self.height)]
        b2Corners = [(box2.x, box2.y), (box2.x+box2.width, box2.y), (box2.x, box2.y+box2.height), (box2.x+box2.width, box2.y+box2.height)]

        if b1Corners[0][0] < b2Corners[0][0] and b1Corners[0][1] < b2Corners[0][1] and b1Corners[3][0] > b2Corners[0][0] and b1Corners[3][1] > b2Corners[0][1]:
            #top left corner of box2 is within box1
            return (self.x+self.width - box2.x)*(self.y+self.height - box2.y)

        elif b1Corners[0][0] < b2Corners[1][0] and b1Corners[0][1] < b2Corners[1][1] and b1Corners[3][0] > b2Corners[1][0] and b1Corners[3][1] > b2Corners[1][1]:
            #top right corner of box2 is within box1 
            return (box2.x+box2.width - self.x)*(self.y+self.height - box2.y)

        elif b1Corners[0][0] < b2Corners[2][0] and b1Corners[0][1] < b2Corners[2][1] and b1Corners[3][0] > b2Corners[2][0] and b1Corners[3][1] > b2Corners[2][1]:
            #bottom left corner of box2 is within box1
            return (self.x+self.width - box2.x)*(box2.y+box2.height - self.y)

        elif b1Corners[0][0] < b2Corners[3][0] and b1Corners[0][1] < b2Corners[3][1] and b1Corners[3][0] > b2Corners[3][0] and b1Corners[3][1] > b2Corners[3][1]:
            #bottom right corner of box2 is within box1
            return (box2.x+box2.width - self.x)*(box2.y+box2.height - self.y)

        else: #no intersection
            if recursive:
                return 0
            else:
                return box2.intersect(self, True)

    def union(self, box2):
        return ((self.height*self.width)+(box2.height*box2.width)) - self.intersect(box2)

class TargetBoundingBox(BoundingBox):
    def __init__(self, x, y, width, height):
        super().__init__(x, y, width, height)
        self.assignedBox = -1