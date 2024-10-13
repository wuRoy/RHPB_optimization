import numpy as np
from pathlib import Path


def sort_adding_command(composition_x: np.ndarray, type_plate: int):
    '''
    To sort the adding command for each composition to maximize the dispensing speed
    '''
    column_num = 12  # 96-well plate have 12 columns, for 384 well-palte this should be changed to 32
    portion_w_zeros = np.empty([0, 2])
    portion_wo_zeros = np.empty([0, 2])
    well_num = np.array(range(1, type_plate + 1))
    if (composition_x.shape[0] < type_plate):
        volume_label = np.column_stack((composition_x, well_num[:composition_x.shape[0]])) 
    else:
        volume_label = np.column_stack((composition_x, well_num))
    volume_label = np.array_split(volume_label, column_num)
    for i in volume_label:
        counter = 0
        for j in i:
            if j[0] == 0:
                portion_w_zeros = np.append(portion_w_zeros, i, axis=0)
                break
            counter = counter + 1
            if counter == i.shape[0]:
                portion_wo_zeros = np.append(portion_wo_zeros, i, axis=0)

    return (portion_w_zeros, portion_wo_zeros)


def tecan_print(
        source_path: Path,
        dest_path: Path,
        num_wells_source_plate: int = 4,
        type_plate: int = 96,
        total_volume: int = 100
):
    '''
    Convert the composition file to .csv that can be read by Tecan.
    The format is: 'Source label','Source position','Destination label','Detination position','volume'
        num_wells_source_plate: the number of wells in the source plate, could be 1,2,4,6,8,12
    '''
    # read composition from csv file
    composition = np.genfromtxt(source_path, delimiter=",")
    composition = composition.round(2)
    volume = composition * total_volume
    volume = volume.round(1)
    command_w_zeros = np.empty([0, 5])
    command_wo_zeros = np.empty([0, 5])

    for c in range(volume.shape[1]):
        counter = 0
        sorted_row_w_zeros, sorted_row_wo_zeros = sort_adding_command(
            composition_x=volume[:, c], type_plate=type_plate)

        # output for parallel dispensing
        for r in range(sorted_row_wo_zeros.shape[0]):
            if sorted_row_wo_zeros[r, 0] != 0:
                command_line = np.array(
                    [f'P{(int(c / num_wells_source_plate)+1)}', 
                     ((c % num_wells_source_plate) * 8 + 1 + counter), 
                     'Dest', 
                     sorted_row_wo_zeros[r, 1], 
                     sorted_row_wo_zeros[r, 0]])
                command_wo_zeros = np.vstack((command_wo_zeros, command_line))
                counter = counter + 1
                if (counter == 8):
                    counter = 0

        # output for individual dispensing
        for r in range(sorted_row_w_zeros.shape[0]):
            if sorted_row_w_zeros[r, 0] != 0:
                command_line = np.array(
                    [f'P{(int(c / num_wells_source_plate) + 1)}', 
                     ((c % num_wells_source_plate) * 8 + 1 + counter), 
                     'Dest',
                     sorted_row_w_zeros[r, 1],
                     sorted_row_w_zeros[r, 0]])
                command_w_zeros = np.vstack((command_w_zeros, command_line))
                counter = counter + 1
                if (counter == 8):
                    counter = 0

    command = np.append(command_wo_zeros, command_w_zeros, axis=0) 

    if num_wells_source_plate == 96:
        command_96 = np.empty([0, 5])
        for r in range(volume.shape[0]):
            for c in range(volume.shape[1]):
                if volume[r, c] != 0:
                    command_line = np.array(
                        [f'P{(int(c / num_wells_source_plate) + 1)}', 
                        (c % num_wells_source_plate + 1), 
                        'Dest',
                        r + 1,
                        volume[r, c]])
                    command_96 = np.vstack((command_96, command_line))                            
        command = command_96

    np.savetxt(dest_path, X=command, fmt='%s', delimiter=',')
