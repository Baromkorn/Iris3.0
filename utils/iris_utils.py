from fast_api.core.iris_setup import  matcher
import time

async def Iris_Matcher(probe_template,gallery_array) :
    start = time.time()
    lowest_HD = 1.0
    i = 0
    closest_template = None
    for gallery_template in gallery_array :
        hamming_distance = matcher.run(probe_template, gallery_template)
        if lowest_HD > hamming_distance :
            lowest_HD = hamming_distance
            closest_template = gallery_template
            index = i
        i+=1
    end = time.time()
    print(f"Execution time of Matcher: {end - start:.4f} seconds")
    return closest_template,lowest_HD,index