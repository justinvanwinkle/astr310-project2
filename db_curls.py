import numpy as np

command = ("curl -o '{},{}' -s 'https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-query?"
           "catalog=ptf_lightcurves&spatial=Box&radius=60&radunits=arcsec&"
           "objstr={}+{}&size=60&outfmt=1'")


for ra in np.arange(320, 340, 0.0166667):
    for dec in np.arange(0, 30, 0.0166667):
        print(command.format(ra, dec, ra, dec))
