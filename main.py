from enum import Enum
from typing import TypeVar, Dict, Generic, List, Tuple
import pandas as pd

# constants
TRANSFER_TIME_IN_MIN: int = 4 # this is the time in minutes that the user takes to change the subway line at the station
AVERAGE_TRAIN_SPEED_IN_kM_PER_H: int = 30

# get data
real_distances = pd.read_csv('./data/real_distances.csv', delimiter=';')
real_distances.set_index(real_distances.columns[0], inplace=True)
real_distances = real_distances.astype(str).apply(lambda x: x.str.replace(',', '.')).astype(float)
real_distances = real_distances.add(real_distances.T, fill_value=0)

direct_distances = pd.read_csv('./data/direct_distances.csv', delimiter=';')
direct_distances.set_index(direct_distances.columns[0], inplace=True)
direct_distances = direct_distances.astype(str).apply(lambda x: x.str.replace(',', '.')).astype(float)
direct_distances = direct_distances.add(direct_distances.T, fill_value=0)

subway_lines = pd.read_csv('./data/subway_lines.csv', header=None, delimiter=';')

# Data structures
# Enums
class StationName(Enum):
    E1 = 'E1'
    E2 = 'E2'
    E3 = 'E3'
    E4 = 'E4'
    E5 = 'E5'
    E6 = 'E6'
    E7 = 'E7'
    E8 = 'E8'
    E9 = 'E9'
    E10 = 'E10'
    E11 = 'E11'
    E12 = 'E12'
    E13 = 'E13'
    E14 = 'E14'

class SubwayLineName(Enum):
    BLUE = 'Azul'
    YELLOW = 'Amarela'
    RED = 'Vermelha'
    GREEN = 'Verde'
    
# Graph
NODE = TypeVar('NODE')
class WeightedGraph(Generic[NODE]):
    def __init__(self) -> None:
        self.graph: Dict[NODE, Dict[NODE, int]] = {}

    def __str__(self) -> str:
        return str(self.graph)

    def add_node(self, node: NODE) -> None:
        if node not in self.graph:
            self.graph[node] = {}

    def add_edge(self, from_node: NODE, to_node: NODE, weight: int) -> None: 
        # adding both ways cause this graph is undirected
        self.graph[from_node][to_node] = weight
        self.graph[to_node][from_node] = weight

    def get_nodes(self) -> List[NODE]:
        return list(self.graph.keys())

# problem specific classes
class StationLine():
    def __init__(self, station: StationName, subway_line: SubwayLineName) -> None:
        self.station = station
        self.subway_line = subway_line

    def __eq__(self, other) -> bool:
        # Define equality comparison
        if isinstance(other, StationLine):
            return self.station == other.station and self.subway_line == other.subway_line
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.station, self.subway_line))

    def __str__(self) -> str:
        return f"Station: {self.station}, Line: {self.subway_line}"
    
    def __repr__(self) -> str:
        return f"StationLine(station={self.station}, subway_line={self.subway_line})"

class SubwayGraph(WeightedGraph[StationLine]):
    def __init__(self) -> None:
        # here i will need to add all the data from the subway map
        super().__init__()
        for _index, row in subway_lines.iterrows():
            line = SubwayLineName[row[0]]
            for i in range(1, len(row)):
                if type(row[i]) == str:
                    station = StationName[row[i]]
                    node = StationLine(station=station, subway_line=line)
                    super().add_node(node)
                    if i > 1:
                        previous_station = StationName[row[i-1]]
                        weight = (real_distances[station.value][previous_station.value]/AVERAGE_TRAIN_SPEED_IN_kM_PER_H)*60
                        super().add_edge(node, StationLine(station=previous_station, subway_line=line), weight)
                    equivalent_stations = self.get_same_station_different_line(node)
                    for equivalent_station in equivalent_stations:
                        super().add_edge(node, equivalent_station, TRANSFER_TIME_IN_MIN)
        
    def get_same_station_different_line(self, station: StationLine) -> List[StationLine]:
        function_return = []
        nodes = super().get_nodes()
        for node in nodes:
            if node.station == station.station and node.subway_line != station.subway_line:
                function_return.append(node)
        return function_return
    
    def heuristic(self, node: StationLine, goal: StationLine) -> float:
        predicted_cost = (direct_distances[node.station.value][goal.station.value]/AVERAGE_TRAIN_SPEED_IN_kM_PER_H)*60
        return predicted_cost

    def find_path(self, start: StationLine, goal: StationLine) -> Tuple[List[StationLine], float]:
        # A* algorithm
        open_set = [(self.heuristic(start, goal), 0, start, [])]
        # 1. cost i think i am going to use to reach where i wanna go
        # 2. real cost until here
        # 3. current node
        # 4. path until here
        visited = set()

        while open_set:
            current_entry = min(open_set, key=lambda x: x[0])
            open_set.remove(current_entry)
            estimated_total_cost, cost_to_reach_current, current, path = current_entry

            if current in visited:
                continue

            path = path + [current]

            if current == goal:
                return path, cost_to_reach_current

            visited.add(current)

            for neighbor, cost in self.graph[current].items():
                new_cost = cost_to_reach_current + cost
                estimated_total_cost = new_cost + self.heuristic(neighbor, goal)
                open_set.append((estimated_total_cost, new_cost, neighbor, path))

        return None, None  # If there is no path


# user input
# start_station: StationName = input("In which station are u? ")
# if start_station not in StationName.__members__:
#     print("Invalid station name")
#     exit()
# goal_station: StationName = input("Which station do u wanna go? ")
# if goal_station not in StationName.__members__:
#     print("Invalid station name")
#     exit()

# create the subway graph
subway = SubwayGraph()
start_node = StationLine(station=StationName.E4, subway_line=SubwayLineName.GREEN)
goal_node = StationLine(station=StationName.E11, subway_line=SubwayLineName.RED)
print(subway.find_path(start=start_node, goal=goal_node))
