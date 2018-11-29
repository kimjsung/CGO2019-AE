#
#   Configuration
#
class Config:
    #
    #
    #
    def __init__(self):
        #
        self.list_representative_problem_size   = []
        self.list_tile_sizes                    = []

        #
        self.list_tensor_C  = []
        self.list_tensor_A  = []
        self.list_tensor_B  = []
        self.list_TB_X      = []
        self.list_TB_Y      = []
        self.list_TB_K      = []
        self.list_REG_X     = []
        self.list_REG_Y     = []
        self.list_GRID_X    = []
        self.list_splits    = []

        #
        self.list_possible_split_cases              = []
        self.list_split_representative_problem_size = []

        #
        self.size_TB_X      = 1
        self.size_TB_Y      = 1
        self.size_TB_K      = 1     # internel indices
        self.size_REG_X     = 1
        self.size_REG_Y     = 1

        #
        self.kernel_full_ext    = True
        self.kernel_full_int    = True

        #
        self.kernel_used_registers  = 0

        #
        self.kernel_comp                    = 0
        self.kernel_arithmetic_intensity    = 0

        #
        self.cost_total         = 0
        self.cost_load_input    = 0
        self.cost_load_output   = 0
        self.cost_store_output  = 0
        self.steps_main_loops   = 0

        #
        self.rank               = -1

        #
        self.m  = 0
        self.n  = 0
        self.k  = 0

        #
        self.num_TBs = 0
        self.num_Estimated_Registers = 0

    #
    def add_split_representative_problem_size(self, split_size):
        for each_pair in split_size:
            self.list_split_representative_problem_size.append(each_pair)

    #
    def add_representative_problem_size(self, representative_problem_size):
        for each_pair in representative_problem_size:
            self.list_representative_problem_size.append(each_pair)

    #
    def add_tile_size(self, list_input_tile_sizes):
        for each_pair in list_input_tile_sizes:
            self.list_tile_sizes.append(each_pair)

    #
    def add_split_index(self, list_split_info):
        for each_idx in list_split_info:
            self.list_splits.append([each_idx[0], each_idx[1], each_idx[2]])

    #
    def add_tensor_C(self, tensor_C):
        for each_info in tensor_C:
            self.list_tensor_C.append(each_info)

    #
    def add_tensor_A(self, tensor_A):
        for each_info in tensor_A:
            self.list_tensor_A.append(each_info)
    
    #
    def del_idx_tensor_A(self, str_target_idx):
        idx_count = 0
        for each_idx in self.list_tensor_A:
            if each_idx == str_target_idx:
                self.list_tensor_A.pop(idx_count)

            idx_count = idx_count + 1
            
    #
    def offset_tensor_A(self, str_target_idx):
        idx_count = 0;
        for each_idx in self.list_tensor_A:
            if each_idx == str_target_idx:
                return idx_count
            
            idx_count = idx_count + 1 

        return -1

    #
    def add_tensor_B(self, tensor_B):
        for each_info in tensor_B:
            self.list_tensor_B.append(each_info)

    #
    def del_idx_tensor_B(self, str_target_idx):
        idx_count = 0
        for each_idx in self.list_tensor_B:
            if each_idx == str_target_idx:
                self.list_tensor_B.pop(idx_count)
            
            idx_count = idx_count + 1

    #
    def offset_tensor_B(self, str_target_idx):
        idx_count = 0
        for each_idx in self.list_tensor_B:
            if each_idx == str_target_idx:
                return idx_count
            
            idx_count = idx_count + 1
        
        return -1

    #
    def add_GRID_X(self, GRID_X):
        for each_info in GRID_X:
            self.list_GRID_X.append(each_info)

    #
    def add_TB_X(self, TB_X):
        for each_info in TB_X:
            self.list_TB_X.append(each_info)

    #
    def add_TB_Y(self, TB_Y):
        for each_info in TB_Y:
            self.list_TB_Y.append(each_info)

    #
    def add_TB_K(self, TB_K):
        for each_info in TB_K:
            self.list_TB_K.append(each_info)

    #
    def add_REG_X(self, REG_X):
        for each_info in REG_X:
            self.list_REG_X.append(each_info)

    #
    def add_REG_Y(self, REG_Y):
        for each_info in REG_Y:
            self.list_REG_Y.append(each_info)

    #
    def print_representative_problem_size(self):
        print ("Representative Problem Size: ", self.list_representative_problem_size)
    #
    def print_split_representative_problem_size(self):
        print ("Split-Representative Problem-Size: ", self.list_split_representative_problem_size)

    #
    def print_splits(self):
        print ("Split Indices: ", self.list_splits)

    #
    def print_TC_Equation(self):
        print ("TC: ", self.list_tensor_C, " = ", self.list_tensor_A, " * ", self.list_tensor_B)

    #
    def print_tensor_C(self):
        print ("Tensor C: ", self.list_tensor_C)

    #
    def print_tensor_A(self):
        print ("Tensor A: ", self.list_tensor_A)
    
    #
    def print_tensor_B(self):
        print ("Tensor B: ", self.list_tensor_B)

    #
    def print_REG_X(self):
        print ("REG_X: ", self.list_REG_X)

    #
    def print_REG_Y(self):
        print ("REG_Y: ", self.list_REG_Y)

    #
    def print_REG(self):
        print ("REG_X: ", self.list_REG_X, ", REG_Y: ", self.list_REG_Y)

    #
    def print_TB_X(self):
        print ("TB_X: ", self.list_TB_X)

    #
    def print_TB_Y(self):
        print ("TB_Y: ", self.list_TB_Y)
    
    #
    def print_TB_K(self):
        print ("TB_K: ", self.list_TB_K)

    #
    def print_TB(self):
        print ("TB_X: ", self.list_TB_X, ", TB_Y: ", self.list_TB_Y, ", TB_K: ", self.list_TB_K)

    #
    def print_GRID_X(self):
        print ("BX_X: ", self.list_GRID_X)

    #
    def print_tile_sizes(self):
        print ("Tile-Sizes: ", self.list_tile_sizes)
    
    #
    def print_kernel_full(self):
        print ("Kernel (Full) Ext: ", self.kernel_full_ext, ", Int: ", self.kernel_full_int)

    #
    def print_arithmetic_intensity(self):
        print ("Kernel--- arithmetic intensity: ", self.kernel_arithmetic_intensity)

    #
    def print_mnk(self):
        print ("m: ", self.m, ", n: ", self.n, ", k: ", self.k)

    #
    def print_numTBs(self):
        print ("The estimated number of TBs: ", self.num_TBs)

    #
    def print_configuration(self, opt=0, str=""):
        if opt == 0:
            if str == "":
                print ("============================================================================")
            else:
                print ("====", str, "========================================================================")
        else:
            if str == "":
                print ("====Picked==================================================================")
            else:
                print ("====Picked=", str, "=================================================================")
        self.print_representative_problem_size()
        self.print_split_representative_problem_size()
        self.print_tile_sizes()
        self.print_TC_Equation()
        self.print_REG()
        self.print_TB()
        self.print_kernel_full()
        self.print_arithmetic_intensity()
        self.print_splits()
        self.print_mnk()
        self.print_numTBs()
        print ("|TB| = ", self.size_TB_X, ", ", self.size_TB_Y, ", |REG| = ", self.size_REG_X, ", ", self.size_REG_Y, "|TB_K| = ", self.size_TB_K)
        print ("Total-Cost: ", self.cost_total, ", # of steps for main-loop: ", self.steps_main_loops)
        print ("============================================================================")
