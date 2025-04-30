import torch as th

class InputPerturbationSolver():
    """
    Input Perturbation solver class.
    """
    def __init__(self, *args, perturbation = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.perturbation = perturbation 

    def preprocess_input_unroll(self, input):
        input = input.as_subclass(th.Tensor)
        perturbation = (self.perturbation * input.abs().max() * 
                        th.randn_like(input))
        return input + perturbation
    
    def extract_data(self, input_pts, output_pts):
        u_start, u_final, t_start, t_end, du = super().extract_data(
            input_pts, output_pts)
        perturbation = (self.perturbation * u_start.abs().max() * 
                        th.randn_like(u_start))
        u_start = u_start + perturbation
        return u_start, u_final, t_start, t_end, du