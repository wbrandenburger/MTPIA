def get_patch_limits(shape, x_limit, y_limit, x_margin, y_margin,):    
   
    patch_limits = []
    amax_limits = []

    p_in_x = 1
    while (p_in_x*x_limit - (p_in_x-1)*x_margin) < shape[1]: p_in_x = p_in_x + 1
    p_in_y = 1
    while (p_in_y*y_limit - (p_in_y-1)*y_margin) < shape[0]: p_in_y = p_in_y + 1

    px_max = 0
    overlapx = [0]
    stepsx = shape[1]/float(p_in_x)
    for px in range(p_in_x):
        px_min = int(stepsx/2 + px*stepsx) - int(x_limit/2)
        if px_min < 0: px_min=0      
        if px_max > 0: overlapx.append(px_max - px_min)
        
        px_max = int(stepsx/2 + px*stepsx) + int(x_limit/2)
        if px_max > shape[1]: px_max = shape[1]
        
        py_max = 0
        overlapy = [0]
        stepsy = shape[0]/float(p_in_y)
        for py in range(p_in_y):
            py_min = int(stepsy/2 + py*stepsy) - int(y_limit/2)
            if py_min < 0: py_min=0
            if py_max > 0: overlapy.append(py_max - py_min)
            py_max = int(stepsy/2 + py*stepsy) + int(y_limit/2)
            if py_max > shape[0]: py_max = shape[0]
            
            patch_limits.append([py_min, py_max, px_min, px_max])
            #print([px_min, px_max, py_min, py_max])    count = count + 1


    overlapx.append(0)
    overlapy.append(0)

    for px in range(p_in_x):
        px_min = int(stepsx/2 + px*stepsx) - int(x_limit/2) + int(overlapx[px]/2)
        if px_min < 0: px_min=0
        px_max = int(stepsx/2 + px*stepsx) + int(x_limit/2) - int(overlapx[px+1]/2 + 0.5)
        if px_max > shape[1]: px_max=shape[1]
        
        for py in range(p_in_y):
            py_min = int(stepsy/2 + py*stepsy) - int(y_limit/2) + int(overlapy[py]/2)
            if py_min < 0: py_min=0
            py_max = int(stepsy/2 + py*stepsy) + int(y_limit/2) - int(overlapy[py+1]/2 + 0.5)
            if py_max > shape[0]: py_max=shape[0]
            
            amax_limits.append([py_min, py_max, px_min, px_max])

    return patch_limits, amax_limits