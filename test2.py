import time
from random import randrange

from masterthesis.logger import Logger, LoggerKeys, DurationFormats, Mapping, Colors

mapping = [
    Mapping(LoggerKeys.START,'Start', '', '{}', 26, None),
    Mapping(LoggerKeys.END, 'End', '', '{}', 26, None),
    Mapping(LoggerKeys.DURATION, 'Tick Duration', '', DurationFormats.TotalSeconds, 7, Colors.LIGHTMAGENTA_EX),
    Mapping(LoggerKeys.AVG_DURATION, 'Avg. Duration', '', DurationFormats.TotalSeconds, 6, Colors.LIGHTGREEN_EX),
    Mapping('test', 'Test', '', '{0:.2f}', 9, None),
    Mapping(LoggerKeys.TOTAL_DURATION, 'Total Duration', '', DurationFormats.MinutesSeconds, 8, Colors.LIGHTYELLOW_EX),   
]

eval_mappings = [
    Mapping(None, '', '', 'Eval at step:'),
    Mapping(LoggerKeys.PREVIOUS_STEP, '', '', color=Colors.LIGHTRED_EX),
    Mapping(None, '', '', 'R1:'),
    Mapping('r1', 'R1', '', color=Colors.LIGHTGREEN_EX, length=7, use_current_step=False),
    Mapping(None, '', '', 'R2:'),
    Mapping('r2', 'R2', '', color=Colors.LIGHTYELLOW_EX, length=7, use_current_step=False),
    Mapping(None, '', '', 'RL:'),
    Mapping('rl', 'RL', '', color=Colors.LIGHTBLUE_EX, length=7, use_current_step=False),
]



if __name__ == "__main__":
    logger = Logger(None, mapping, header_frequency=3)
    for i in range(10):
        if i%3 == 0:            
            logger.add_value('r1', .45)
            logger.add_value('r2', .25)
            logger.add_value('rl', 0.3)
            logger.print_line(eval_mappings)
            print()
        time.sleep(randrange(5, 10))
        if i%2 == 0:
            logger.add_value('test', i)
        
        logger.step((i+1)*1000)