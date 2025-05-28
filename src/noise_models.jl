export loss_gaussian, loss_poisson, loss_anscombe, loss_poisson_pos, loss_anscombe_pos

logmul_safe(dat, val) = ifelse.(val == zero(eltype(val)) && dat == zero(eltype(dat)), zero(eltype(val)), dat*log(val))
clip_pos(val) = max(zero(eltype(val)), val)

loss_gaussian(data, fwd, bg=0) = sum(abs2.(data.-fwd))
loss_poisson(data, fwd, bg=eltype(data)(0)) = sum((fwd.+bg) .- logmul_safe.(data.+bg,fwd.+bg))
loss_poisson_pos(data, fwd, bg=eltype(data)(0)) = sum(clip_pos.(fwd.+bg) .- logmul_safe.(clip_pos.(data.+bg), clip_pos.(fwd.+bg)))
loss_anscombe(data, fwd, bg=eltype(data)(0)) = sum(abs2.(sqrt.(data.+bg) .- sqrt.(fwd.+bg)))
loss_anscombe_pos(data, fwd, bg=eltype(data)(0)) = sum(abs2.(sqrt.(clip_pos.(data.+bg)) .- sqrt.(clip_pos.(fwd.+bg))))

