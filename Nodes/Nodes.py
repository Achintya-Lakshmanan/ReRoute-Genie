from object_tracking import test_node

def final_nodes() :
    test_locations = {'L1': (40.819688, -73.915091),
                    'L2': (40.815421, -73.941761),
                    'L3': (40.764198, -73.910785),
                    'L4': (40.768790, -73.953285),
                    'L5': (40.734851, -73.952950),
                    'L6': (40.743613, -73.977998),
                    'L7': (40.745313, -73.993793),
                    'L8': (40.662713, -73.946101),
                    'L9': (40.703761, -73.886496),
                    'L10': (40.713620, -73.943076),
                    'L11': (40.725212, -73.809179)
                    }

    for i in test_locations:
        if test_node.no_congestion_video() == True:
            continue
        else :
            test_locations.pop(i) # Multiplicative factor
    
    return test_locations