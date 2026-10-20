using Test

include(joinpath(@__DIR__, "common", "model_checkpoints.jl"))
using .ModelCheckpoints

@testset "trained model checkpoints" begin
    directory = mktempdir()
    path = joinpath(directory, "cell.checkpoint.jls")
    identity = checkpoint_identity(17, :nn, 3, "f"^64, "s"^64)
    payload = (ps=(weights=[1.0, 2.0],), st=(mode=:test,),
               fit_time_s=123.5, final_loss=0.25)

    @test write_model_checkpoint(path, identity, payload) == path
    loaded = load_model_checkpoint(path, identity)
    @test loaded == payload
    @test loaded.fit_time_s == 123.5

    @test load_model_checkpoint(path,
        checkpoint_identity(17, :gp, 3, "f"^64, "s"^64)) === nothing
    @test load_model_checkpoint(path,
        checkpoint_identity(17, :nn, 4, "f"^64, "s"^64)) === nothing
    @test load_model_checkpoint(path,
        checkpoint_identity(17, :nn, 3, "x"^64, "s"^64)) === nothing
    @test load_model_checkpoint(path,
        checkpoint_identity(17, :nn, 3, "f"^64, "x"^64)) === nothing

    write(path, "not a serialized checkpoint")
    @test load_model_checkpoint(path, identity) === nothing
end
