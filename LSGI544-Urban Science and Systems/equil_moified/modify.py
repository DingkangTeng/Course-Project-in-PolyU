import copy
from xml.etree.cElementTree import Element, ElementTree

class modify:
    __slots__ = ["tree", "root", "path", "doctype"]

    def __init__(self, path: str):
        self.tree = ElementTree()
        self.tree.parse(path)
        self.root = self.tree.getroot()
        self.path = path

        # Read doctype
        with open(path, 'r') as f:
            lines = f.readlines()
        self.doctype = ""
        for line in lines:
            if line.startswith("<!DOCTYPE"):
                self.doctype = line.strip()
                break

    def write(self) -> None:
        with open(self.path, 'wb') as f:
            f.write(f'<?xml version="1.0" encoding="utf-8"?>\n'.encode('utf-8'))
            f.write(f'{self.doctype}\n'.encode('utf-8'))
            self.tree.write(f, encoding='utf-8', xml_declaration=False)
    
    def findall(self, path: str) -> list[Element]:
        return self.tree.findall(path)
    
    def creatNode(self, tag: str, propertyMap: dict, content: str = "") -> Element:
        element = Element(tag, propertyMap)
        element.text = content
        
        return element

    def addChildNode(self, nodelist: list[Element], element: Element) -> None:
        for node in nodelist:
            node.append(element)
        
        return
    
    def rootAppend(self, element: Element) -> None:
        self.addChildNode([self.root], element)

        return

def swapAttrib(attribDict: dict, swapA: str, swapB: str) -> None:
    A = attribDict.get(swapA)
    B = attribDict.get(swapB)
    attribDict[swapA] = B
    attribDict[swapB] = A

    return

class changModel:
    __slots__ = ["totalNum", "car", "pt", "bicycle"]
    
    def __init__(self, totalNum: int):
        self.totalNum = totalNum
        self.car = 0
        self.pt = 0
        self.bicycle = 0

    def detreminModel(self, car: float, pt: float, bicycle: float) -> str:
        if self.totalNum * car > self.car:
            self.car += 1
            return "car"
        elif self.totalNum * bicycle > self.bicycle:
            self.bicycle += 1
            return "bike"
        elif self.totalNum * pt > self.pt:
            self.pt += 1
            return "pt"
        else:
            return "walk"

if __name__ == "__main__":
    # # Modify the network
    # a = modify("network.xml")
    # links = a.findall("links")
    # for node in links:
    #     links = node.findall("link")
    #     linkNum = len(node)
    #     for link in links:
    #         linkNum += 1
    #         # Get the original link
    #         attribDict = link.attrib.copy()
    #         # Swap the start and end points
    #         swapAttrib(attribDict, "from", "to")
    #         attribDict["id"] = str(linkNum) # Ensure id is a string
    #         # Add new link
    #         node.append(a.creatNode("link", attribDict))
    
    # a.write()

    # # Modify the plans
    # b = modify("plans100.xml")
    # people = b.findall("person")
    # peopleNum = len(people)
    # peopleModel = changModel(peopleNum)
    # for person in people:
    #     personPlans = person.findall("plan")
    #     for personPlan in personPlans:
    #         legs = personPlan.findall("leg")
    #         model = peopleModel.detreminModel(1, 0, 0)
    #         for leg in legs:
    #             leg.set("mode", model)
    # b.write()

    # Double agents
    c = modify("plans100_double.xml")
    people = c.findall("person")
    peopleNum = len(people)
    for person in people:
        # Duplicates how 1 time
        for i in range(1):
            peopleNum += 1
            newPerson = copy.deepcopy(person)
            newPerson.set("id", str(peopleNum)) # Ensure id is a string
            c.rootAppend(newPerson)
    c.write()
            