use Math::Complex;

$LSD=10;

@omega = 
( -1.3621887071757046,
-0.5198725907919344,
-0.2060321767089241,
-0.10132532424932489,
0.06924026783865815+i*0.05673404746065078,
-0.06924026783865815+i*(-0.05673404746065078),
-0.12282664428523404,
-0.33072408812180654,
-0.7888980515057518,
-1.1118735302951177,
);

open(O,">doext_arg.vml");
print O <<'EOT';
class DoArgExt doext_arg = {
double twist_bc_x = 0.0
double twist_bc_y = 0.0
double twist_bc_z = 0.0
double twist_bc_t = 0.0
StartConfType start_u1_conf_kind = START_CONF_FILE
uint64 start_u1_conf_load_addr = 0
string start_u1_conf_filename = "/home/izubuchi/src/lanczos-single/cps_pp/work/complex-mobius/ncu1f.0.ieee"
int start_u1_conf_alloc_flag = 6
int mult_u1_conf_flag = 0
int save_stride = 2
int trajectory = 0
double mobius_b_coeff =   -222.0
double mobius_c_coeff =   -222.0
EOT



#      // b[s]=c[s]+1
#      // 1/omega[s] = b[s]+c[s]
#      // =>  b[s] = (1-1/omega[s])/2
#      //     c[s] = b[s]-1

printf( O "Array zmobius_b_coeff[%d] = {\n", $LSD*2);

for($s=0;$s<$LSD;++$s){
    $b = (1.0 - 1.0/$omega[$s])/2.0;
    printf(O "double zmobius_b_coeff[%d] = %.16e\n", $s*2, Re($b));
    printf(O "double zmobius_b_coeff[%d] = %.16e\n", $s*2+1,Im($b));
}

print O"}\n";

printf( O "Array zmobius_c_coeff[%d] = {\n", $LSD*2);

for($s=0;$s<$LSD;++$s){
    $b = (1.0 - 1.0/$omega[$s])/2.0;
    $c = $b-1.0;
    printf(O "double zmobius_c_coeff[%d] = %.16e\n", $s*2, Re($c));
    printf(O "double zmobius_c_coeff[%d] = %.16e\n", $s*2+1,Im($c));
}

print O"}\n";
print O"}\n";


close O;


