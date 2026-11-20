import importlib.util, os
from falkorchat import repository as R
spec = importlib.util.spec_from_file_location("falkorchat._repo_mut", os.environ["MUT_REPO"], submodule_search_locations=None)
m = importlib.util.module_from_spec(spec); m.__package__ = "falkorchat"; import sys; sys.modules[spec.name] = m; spec.loader.exec_module(m)
for n in ("post_first_message", "post_subsequent_message"):
    setattr(R.Repository, n, getattr(m.Repository, n))
print("MUTANT LOADED")
