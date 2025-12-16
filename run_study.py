
import pandas as pd
from pommes.io.build_input_dataset import *
from pommes.model.data_validation.dataset_check import check_inputs
from pommes.io.save_solution import save_solution
from pommes.model.build_model import build_model
import warnings
import time
from datetime import timedelta

warnings.filterwarnings("ignore")

all_areas=['AL', 'AT', 'BA', 'BE', 'BG', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES',
       'FI', 'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MD', 'ME',
       'MK', 'MT', 'NL', 'NO', 'PL', 'PT', 'RO', 'RS', 'SE', 'SI', 'SK','UK']

if __name__ == "__main__":
        title=f"\033[1m Running POMMES-EU \033[0m"

        print(title)



        areas=["BE","FR"]#,"DE","NL","UK","ES","IT","CH","DK"]
        year_op=[2050]
        solver = "highs"  # gurobi, cplex, mosek, highs
        threads = 4

        if areas==["all"]:
            areas=all_areas


        add=""
        if len(year_op)<4*3:
            for i in year_op:
                add+="-"+str(i)
        suffix = f"{len(areas)}-nodes"+add
        if len(areas)==1:
            suffix=f"{areas[0]}"+add
        elif len(areas)==2:
            suffix = f"{areas[0]}-{areas[1]}" + add

        print(suffix)
        print(year_op,areas)
        output_folder = f"output/{suffix}"

        start = time.time()
        print("\033[1m Input data load and pre-processing \033[0m")
        config = read_config_file(file_path="config.yaml")
        config["input"]["path"] = "data"
        config["coords"]["area"]["values"]=areas
        config["coords"]["year_op"]["values"] = year_op


        ####Link adjustment
        if config["add_modules"]["transport"] :
            areas=config["coords"]["area"]["values"]
            all_links=pd.read_csv(config["input"]["path"]+"/transport_link.csv",sep=";").link.unique()
            links = []

            for link in all_links:
                pos = ""
                i = 0
                while pos != "-":
                    pos = link[i]
                    i += 1
                area_from = link[:i - 1]
                area_to = link[i:]
                if area_to in areas and area_from in areas:
                    links.append(link)
            if len(links) >= 1:
                config["coords"]["link"]["values"] = links
            else:
                config["add_modules"]["transport"]=False
        print("\t Transport activated:", config["add_modules"]["transport"])

        model_parameters = build_input_parameters(config)
        model_parameters = check_inputs(model_parameters)

        print("\033[1m Model building \033[0m")
        start_build = time.time()
        model = build_model(model_parameters)



        elapsed_time = time.time() - start_build
        print("\t Model building took {}".format(timedelta(seconds=elapsed_time)))

        print("\033[1m Model solving \033[0m")
        if solver == "gurobi":
            model.solve(solver_name="gurobi", #progress=True,
                        io_api='direct',
                        threads=threads, method=2,
                        nodefilestart=0.1, presparsify=2, presolve=2,#memlimit=8,
                        logtoconsole=1, outputflag=1
                        )
        elif solver=="cplex":
            solver_options={"threads":threads,"preprocessing.presolve":1,"lpmethod":4,
                            # 'barrier.convergetol':5e-5
                            }
            model.solve(solver_name="cplex",**solver_options) #parameters={"presolve":1} epfi=1e-5,
        elif solver=="highs":
            model.solve(solver_name="highs",presolve='on',solver='hipo',run_crossover='choose')

        else:
            model.solve(solver_name=solver, threads=threads)

        converge = True
        print(model.termination_condition )
        if model.termination_condition not in ["optimal","suboptimal"]:
            try:
                print("\t Searching for infeasabilities")
                model.compute_infeasibilities()
                model.print_infeasibilities()
                converge = False
            except:
                converge = False
        if converge:
            print("\033[1m Results export \033[0m")
            save_solution(
                model=model,
                output_folder=output_folder,
                save_model=False,
                export_csv=False,
                model_parameters=model_parameters,
            )

            elapsed_time = time.time() - start
            print("Process took {}".format(timedelta(seconds=elapsed_time)))

        del model
        del model_parameters