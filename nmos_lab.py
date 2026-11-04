# Classroom NMOSFET simulation
# Units: cm, V, s, carrier density in cm^-3. 2D currents use DEVSIM native normalization.
import time
import numpy as np
import devsim as ds
from devsim.python_packages.simple_physics import GetContactBiasName
from devsim.python_packages.ramp import rampbias
for _precision_mode in ("extended_solver", "extended_model", "extended_equation"):
    ds.set_parameter(name=_precision_mode, value=True)
REL_ERROR = 1e-7  # Try 1e-7 as a numerical sensitivity check.

def _initialize():
    # Copyright 2013 DEVSIM LLC
    #
    # SPDX-License-Identifier: Apache-2.0
    
    from devsim.python_packages.simple_physics import (
        GetContactBiasName,
        SetOxideParameters,
        SetSiliconParameters,
        CreateSiliconPotentialOnly,
        CreateSiliconPotentialOnlyContact,
        CreateSiliconDriftDiffusion,
        CreateSiliconDriftDiffusionAtContact,
        CreateOxidePotentialOnly,
        CreateSiliconOxideInterface,
    )
    from devsim.python_packages.ramp import rampbias, printAllCurrents
    from devsim import (
        element_from_edge_model,
        get_contact_list,
        get_region_list,
        node_model,
        set_node_values,
        set_parameter,
        solve,
        write_devices,
    )
    from devsim.python_packages.model_create import CreateSolution
    
    
    import importlib, sys
    # Clear DEVSIM state before rebuilding after a kernel-cell rerun.
    import devsim as ds
    for old_device in ds.get_device_list():
        ds.delete_device(device=old_device)
    if "mos2d" in ds.get_mesh_list():
        ds.delete_mesh(mesh="mos2d")
    if "gmsh_mos2d_create" in sys.modules:
        importlib.reload(sys.modules["gmsh_mos2d_create"])
    else:
        import gmsh_mos2d_create
    
    device = "mos2d"
    silicon_regions = ("gate", "bulk")
    oxide_regions = ("oxide",)
    regions = ("gate", "bulk", "oxide")
    interfaces = ("bulk_oxide", "gate_oxide")
    
    for i in regions:
        CreateSolution(device, i, "Potential")
    
    for i in silicon_regions:
        SetSiliconParameters(device, i, 300)
        CreateSiliconPotentialOnly(device, i)
    
    for i in oxide_regions:
        SetOxideParameters(device, i, 300)
        CreateOxidePotentialOnly(device, i, "log_damp")
    
    ### Set up contacts
    contacts = get_contact_list(device=device)
    for i in contacts:
        tmp = get_region_list(device=device, contact=i)
        r = tmp[0]
        print("%s %s" % (r, i))
        CreateSiliconPotentialOnlyContact(device, r, i)
        set_parameter(device=device, name=GetContactBiasName(i), value=0.0)
    
    for i in interfaces:
        CreateSiliconOxideInterface(device, i)
    
    solve(type="dc", absolute_error=1e-12, relative_error=1e-10, maximum_iterations=50)
    
    # Intermediate VTK export omitted for speed.
    
    for i in silicon_regions:
        CreateSolution(device, i, "Electrons")
        CreateSolution(device, i, "Holes")
        set_node_values(
            device=device, region=i, name="Electrons", init_from="IntrinsicElectrons"
        )
        set_node_values(device=device, region=i, name="Holes", init_from="IntrinsicHoles")
        CreateSiliconDriftDiffusion(device, i, "mu_n", "mu_p")
    
    for c in contacts:
        tmp = get_region_list(device=device, contact=c)
        r = tmp[0]
        CreateSiliconDriftDiffusionAtContact(device, r, c)
    
    solve(type="dc", absolute_error=1.0e30, relative_error=REL_ERROR, maximum_iterations=60)
    

def build():
    """Build the original fixed mesh, electrostatics, then drift diffusion."""
    started = time.perf_counter()
    _initialize()
    return time.perf_counter() - started

def bias(contact, value, step=0.1):
    """Continue from the current solution; reduce steps on convergence failure."""
    rampbias("mos2d", contact, float(value), step, 0.0001, 60,
             REL_ERROR, 1e30, lambda device: None)
    actual=ds.get_parameter(device="mos2d", name=GetContactBiasName(contact))
    if abs(actual-value)>1e-8:
        raise RuntimeError("Requested endpoint was not reached")

def currents():
    """Return signed total terminal currents and a normalized KCL residual."""
    result={}
    for c in ds.get_contact_list(device="mos2d"):
        result[c]=sum(ds.get_contact_current(device="mos2d",contact=c,equation=e)
                      for e in ("ElectronContinuityEquation","HoleContinuityEquation"))
    scale=sum(abs(i) for i in result.values())
    result["kcl_relative"]=abs(sum(result.values()))/max(scale,1e-30)
    return result

def sweep(contact, values, step=0.1):
    """Record one row per requested bias. Internal continuation points are not rows."""
    start=time.perf_counter(); rows=[]
    for value in values:
        bias(contact,value,step)
        rows.append({"bias":float(value),**currents()})
    print(f"Sweep time: {time.perf_counter()-start:.2f} s")
    return rows

def surface_profile():
    """Sample near-interface bulk nodes. This is not an interpolation or a cut line."""
    x=np.asarray(ds.get_node_model_values(device="mos2d",region="bulk",name="x"))
    y=np.asarray(ds.get_node_model_values(device="mos2d",region="bulk",name="y"))
    n=np.asarray(ds.get_node_model_values(device="mos2d",region="bulk",name="Electrons"))
    psi=np.asarray(ds.get_node_model_values(device="mos2d",region="bulk",name="Potential"))
    mask=np.abs(y)<1e-12
    order=np.argsort(x[mask])
    return x[mask][order]*1e4,n[mask][order],psi[mask][order]
