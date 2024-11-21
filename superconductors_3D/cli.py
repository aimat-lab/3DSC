import argparse
from superconductors_3D.generate_3DSC import main as generate_3DSC
from pathlib import Path

init_cli_output = rf"""===============================================================================

                  ______    ______      ______      ______  
                 / ____ `. |_   _ `.  .' ____ \   .' ___  | 
                 `'  __) |   | | `. \ | (___ \_| / .'   \_| 
                 _  |__ '.   | |  | |  _.____`.  | |        
                | \____) |  _| |_.' / | \____) | \ `.___.'\ 
                 \______.' |______.'   \______.'  `.____ .' 
                                                      
                    Authors - Timo Sommer, Pascal Friederich
==============================================================================="""

def check_n_args(args, n):
    if len(args) != n:
        raise ValueError(f'Expected {n} path, got {len(args)} arguments.')

def parse_input_parameters():
    desc = f"""3DSC code for generating 3D structures of superconductors.
    Usage: superconductors_3D --database <database> --n-cpus <n_cpus> --datadir <datadir>
    """
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--database', '-d', dest='database', type=str)
    parser.add_argument('--n-cpus', '-n', dest='n_cpus', type=int)
    parser.add_argument('--datadir', '-dd', dest='datadir', type=str)
    args = parser.parse_args()
    return args

def main():

    args = parse_input_parameters()

    print(init_cli_output)

    generate_3DSC(
                    crystal_database=args.database,
                    n_cpus=args.n_cpus,
                    datadir=args.datadir
    )


if __name__ == '__main__':
    database = 'ICSD'  # 'MP' or 'ICSD'
    n_cpus = 2  # Number of CPUs to use for parallelization
    datadir = Path('/Users/timosommer/Downloads/test_3DSC/data2')  # Path to the data directory

    args = parse_input_parameters()
    database = args.database if not args.database is None else database
    n_cpus = args.n_cpus if not args.n_cpus is None else n_cpus
    datadir = args.datadir if not args.datadir is None else datadir

    generate_3DSC(
                    crystal_database=database,
                    n_cpus=n_cpus,
                    datadir=datadir
    )
