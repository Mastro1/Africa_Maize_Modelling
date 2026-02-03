
try:
    from pyrealm.core import hygro
    print(f"Has calc_vp_sat: {hasattr(hygro, 'calc_vp_sat')}")
    print(f"Has convert_vp_to_vpd: {hasattr(hygro, 'convert_vp_to_vpd')}")
except Exception as e:
    print(e)
