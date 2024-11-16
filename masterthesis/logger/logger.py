from datetime import datetime
from enum import Enum
from typing import Any, List, Tuple, Union, Dict
from math import floor

from torch.utils.tensorboard import SummaryWriter
from colorama import Style, Fore

Colors = Fore

class LoggerKeys(Enum):
    STEP = 'step'
    START = 'start'
    END = 'end'
    DURATION = 'duration'
    AVG_DURATION = 'avg_duration'
    TOTAL_DURATION = 'total_duration'
    PREVIOUS_STEP = 'previous_step'

class DurationFormats():
    DaysHoursMinutes = '{days:02.0f}d {hours:02.0f}:{minutes:02.0f}'
    HoursMinutesSeconds = '{fl_total_hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}'
    MinutesSeconds = '{fl_total_minutes:02.0f}:{seconds:02.0f}'
    TotalMinutes = '{total_minutes:.2f}'
    TotalSeconds = '{total_seconds:.2f}'

class Mapping():
    def __init__(self, key: str, name:str, tensorName:str, format:str = '{}', length:int = 0, color:str = None, use_current_step:bool=True) -> None:
        self.key = key
        self.name = name
        self.writerName = tensorName
        self.format = format
        self.length = length
        self.color = color
        self.current_step = use_current_step

class Logger:
    def __init__(self, writer: SummaryWriter, field_mapping: List[Mapping], header_frequency:int = 3, avg_window:int = -1) -> None:
        self.logs = []
        self.writer = writer
        self.field_mapping = field_mapping
        self.frequency = header_frequency
        self.avg_window = avg_window
        self._init_step(0)

    
    def step(self, step: int, dump:bool = True) -> None:
        if len(self.logs) > 0:
            self.logs[-1][LoggerKeys.STEP] = step
            self.logs[-1][LoggerKeys.END] = datetime.now()
            duration = int((self.logs[-1][LoggerKeys.END] - self.logs[-1][LoggerKeys.START]).total_seconds())
            self.logs[-1][LoggerKeys.DURATION] = self._get_duration(duration)
            total_duration = int((self.logs[-1][LoggerKeys.END] - self.logs[0][LoggerKeys.START]).total_seconds())
            self.logs[-1][LoggerKeys.TOTAL_DURATION] = self._get_duration(total_duration)
            sum = 0
            if self.avg_window == -1:
                count = len(self.logs)
                for log in self.logs:
                    sum += log[LoggerKeys.DURATION]['total_seconds']
            else:
                count = min(len(self.logs), self.avg_window)
                for i in range(count):
                    sum += self.logs[-(i+1)][LoggerKeys.DURATION]['total_seconds']
            avg_duration = (sum/count)
            self.logs[-1][LoggerKeys.AVG_DURATION] = self._get_duration(avg_duration)
            if dump:
                self._dump()
        
        self._init_step(step)

    def _get_duration(self, duration:float) -> Dict[str, Union[int, float]]:
        return {
                'days': int((duration / 3600) / 24),
                'hours':int((duration / 3600) % 24), 
                'total_hours': (duration / 3600),
                'fl_total_hours': floor(duration / 3600),
                'minutes':int((duration / 60) % 60),
                'total_minutes': duration/60,
                'fl_total_minutes': floor(duration/60),
                'seconds':int(duration % 60),
                'total_seconds': duration,
                'fl_total_seconds': floor(duration)
        }
        
        
    def _init_step(self, step: int):
        self.logs.append({})
        self.logs[-1][LoggerKeys.START] = datetime.now()
        self.logs[-1][LoggerKeys.PREVIOUS_STEP] = step

            
    def add_value(self, key: str, value: Any) -> None:
        self.logs[-1][key] = value 

    def _print_header(self):
        str_array = []
        for map in self.field_mapping:
            str_array.append(f'{{0:<{max(map.length, len(map.name))}}}'.format(map.name))
        print('\t'*2, end='')
        for name in str_array:
            print(name, end=' ')
        print()
        
        pass

    def print_line(self, mappings: List[Mapping], step: int = None,  writeTensorboard: bool = True):
        self._dump_line(mappings, self.logs[-1], writeTensorboard and (self.writer is not None))

    def _dump(self):
        if (len(self.logs)-1) % self.frequency == 0:
            self._print_header()
        print(f"tick {len(self.logs)}\t{Fore.LIGHTRED_EX}{self.logs[-1][LoggerKeys.STEP]}{Style.RESET_ALL}", end='\t')
        self._dump_line(self.field_mapping, self.logs[-1], (self.writer is not None))
    
    def _dump_line(self, mappings: List[Mapping], values: Dict[str, Any], writeTensorboard: bool = False) -> None:
        str_array = []
        for map in mappings:
            # Code for when key not present!
            try:
                if map.key in [LoggerKeys.DURATION, LoggerKeys.TOTAL_DURATION, LoggerKeys.AVG_DURATION]:
                    value = map.format.format(**(values[map.key]))
                elif map.key is None or map.key == '':
                    value = map.format.format('')
                else:
                    value = map.format.format(values[map.key])
            except KeyError:
                    value = ''
            value = f'{{0:<{max(map.length, len(map.name))}}}'.format(value)
            if map.color is not None:
                value = f'{map.color}{value}{Style.RESET_ALL}'
            str_array.append(value)
            if writeTensorboard and (map.writerName is not None and map.writerName != ''):
                self.writer.add_scalar(map.writerName, 
                                        values[map.key], 
                                        values[LoggerKeys.STEP] if map.current_step else values[LoggerKeys.PREVIOUS_STEP])

        for value in str_array:
            print(value, end=' ')
        
        print()


        

