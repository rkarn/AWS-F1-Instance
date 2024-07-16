`timescale 1ns / 1ps

module avg_pooling(
    input clk,
    input pool_en,
    input [7:0] in1,
    input [7:0] in2,
    input [7:0] in3,
    input [7:0] in4,
    output [7:0] out,
    output  pool_done
    );
    
    reg [15:0] pool_out;
    
    always @(posedge clk) begin
        if(pool_en == 1) begin
            pool_out <= (in1+in2+in3+in4)>> 2; //Calculate average
        end
        
    end
    
    assign out = pool_out[7:0];
    assign pool_done = (pool_out==(in1+in2+in3+in4)>> 2)? 1:0;

endmodule



module dense_layer # (parameter NEURON_NB=32, IN_SIZE=196, WIDTH=8)(
    input clk,
    input layer_en,
    input reset,
    input signed[2*WIDTH-1:0] in_data [0:IN_SIZE-1],
    input signed[WIDTH-1:0] weights [0:NEURON_NB-1][0:IN_SIZE-1],
    input signed[WIDTH-1:0] biases [0:NEURON_NB-1],
    output signed[4*WIDTH-1:0] neuron_out [0:NEURON_NB-1],
    output layer_done
    );
    
    reg [0:NEURON_NB-1] neuron_done;
    reg done = 0;
    
    neuron #(.IN_SIZE(IN_SIZE), .WIDTH(WIDTH)) dense_neuron[0:NEURON_NB-1] (.clk(clk), .en(layer_en), .reset(reset), 
                                                                            .in_data(in_data), .weight(weights), .bias(biases), 
                                                                            .neuron_out(neuron_out), .neuron_done(neuron_done)); // Neuron submodules
    always @(posedge clk) begin
        if(neuron_done == '1) begin //All neurons done
            done <= 1;
        end
    end
    
    assign layer_done = done;

    
endmodule


module dense_layer1(
    input clk,
    input enable,
    input reset,
    input signed [15:0] pooled_img [0:195],
    output signed [15:0] layer_out [0:31],
    output layer_done
    );
    
    reg signed [31:0] dense1_res [0:31];
    reg signed [15:0] relu_res [0:31];
    
    //Biases and weights
    localparam signed [7:0] B_ARRAY_L2 [0:31] = '{ 21, -2, -5, 6, 12, -16, 6, 1, 17, 8, 3, 5, -23, 17, 8, 5, 5, 22, 8, 8, 1, 6, -9, 9, 15, 20, -13, -5, 2, 7, 12, 24 };
    
    localparam signed [7:0] W_ARRAY_L2 [0:31] [0:195] = '{
    { 7, -1, 9, 12, 17, 36, 29, 35, 33, 44, 25, 23, -8, 7, 0, 11, -10, -4, -6, 5, 5, 28, 27, 29, 12, 28, 1, 9, -1, -11, -41, -26, -8, -22, -22, 6, -1, 12, 16, 14, -11, 12, -8, -31, -31, -2, -21, -7, 11, 5, 1, -2, -26, 1, 4, -14, -5, -4, -44, -20, -2, 6, 20, 10, -3, -31, -26, -28, -5, -18, -3, -15, -44, 1, -1, -12, 1, 14, -16, -17, -1, 2, 28, -28, 7, 2, -16, 9, -7, -8, -1, -5, -4, 7, 10, 12, 25, -41, 10, 4, 4, -2, 0, 8, 6, -12, -26, -7, 3, 7, 31, -18, -11, 9, 6, -19, 9, 29, 6, -9, -17, -18, -1, -3, 24, -25, 15, -7, -15, -12, 14, 12, -5, -6, 5, -7, 4, 5, 23, -25, 3, 0, 6, -1, 21, 32, 13, 14, 17, 10, 5, 13, -8, 11, 2, -1, 10, 18, 10, 16, 14, 14, 11, 1, -20, -23, -24, 20, 0, -28, -2, -13, -9, 1, -16, -29, -32, -17, -37, -43, -6, 18, -1, -1, -29, -39, -56, -33, -52, -46, -56, -41, -33, -16, -5, 1 },
        { -6, -9, 2, -6, 2, .....  ;
    
    wire dense1_en = enable;
    reg dense1_done = 0;

    
    dense_layer #(.NEURON_NB(32),.IN_SIZE(196), .WIDTH(8)) dense_layer1(.clk(clk), .layer_en(dense1_en), .reset(reset),
                                                                        .in_data(pooled_img), .weights(W_ARRAY_L2), .biases(B_ARRAY_L2),
                                                                        .neuron_out(dense1_res), .layer_done(dense1_done)); //Dense layer
    
    relu relu_activation[31:0] (.data_in(dense1_res), .data_out(relu_res)); //ReLu activation
    
    assign layer_out = relu_res;               
    assign layer_done = dense1_done;

endmodule



module dense_layer2(
    input clk,
    input enable,
    input reset,
    input signed [15:0] in_data [0:31],
    output signed [15:0] layer_out [0:9],
    output layer_done
    );
    
    reg signed [31:0] dense2_res [0:9];
    reg signed [15:0] relu_res [0:9];
  
    //Biases and weights
    localparam signed [7:0] B_ARRAY_L3 [0:9] = '{ -11, 10, 14, -23, -1, 8, 0, 13, 1, -11 };
    
    localparam signed [7:0] W_ARRAY_L3 [0:9] [0:31] = '{
    { 33, -57, -6, -23, -11, -48, -53, 46, -54, -24, 4, 9, -2, -40, -29, 20, -6, 45, -85, 2, -57, 45, 40, -21, -23, -2, -105, -19, -40, 28, -7, 15 },
    { 30, 34, 29, -19, -12, -47, 50, -69, 64, -25, 38, 35, -43, 19, -17, -61, 62, -31, -30, -69, 43, -67, -60, 5, 46, -1, -50, -64, 39, -39, -28, -28 },
    { 2, -48, 30, 33, 69, -11, 28, -1, 28, 25, -22, 58, -11, -120, -6, -28, -72, -12, -33, -4, 1, 81, 21, -22, 24, -100, -59, -59, -20, 27, 37, 39 },
    { -37, 61, -6, -19, -5, 60, 14, 8, 28, -29, -106, 53, 32, -18, -27, -16, -1, -45, 43, 40, 18, -28, 25, -37, -7, -23, 52, -45, 16, -28, 6, -53 },
    { -9, 57, 20, 19, -96, -25, -47, 19, 43, -19, 8, -68, -37, 37, 28, -8, 13, -15, 17, -40, -33, 2, -100, 35, 5, -52, 12, 34, -75, -92, 30, 17 },
    { 16, -8, -127, -69, 28, -82, 29, -47, -7, -19, -11, -55, -10, 37, -46, -14, 26, -1, 39, 35, -37, -5, 10, -20, -35, 84, 55, 13, 37, 25, -5, -36 },
    { 40, 29, -74, -1, -16, 71, -91, -13, -92, 24, 32, -15, -5, -12, -73, -70, 0, -25, -85, 38, -63, 2, -53, 36, 39, 4, -93, 24, 26, 19, 12, 37 },
    { -40, -42, -10, 45, 26, 8, 25, 8, -33, -40, -32, 34, -40, -38, 32, 19, 42, -20, -18, 41, 21, -41, 24, 7, -25, 37, -93, -32, -56, -52, -48, 55 },
    { 4, -73, 17, -34, 17, 0, -34, -9, -50, 10, 29, -75, 37, 10, -19, 0, -63, -1, 12, 2, 9, -63, 7, 40, -12, -54, -17, 26, -10, -5, -7, -89 },
    { -82, -31, 23, -61, -34, 18, -52, -24, -1, 12, 29, -74, -11, 14, 34, 33, 23, -13, -21, -98, -48, 32, 26, -35, 7, 3, 68, 14, -25, -11, -86, 27 }
    };
    
    wire dense2_en = enable;
    reg dense2_done = 0;

    
  dense_layer #(.NEURON_NB(10),.IN_SIZE(32), .WIDTH(8)) dense_layer2(.clk(clk), .layer_en(dense2_en), .reset(reset),
                                                                     .in_data(in_data), .weights(W_ARRAY_L3), .biases(B_ARRAY_L3),
                                                                     .neuron_out(dense2_res), .layer_done(dense2_done)); //Dense layer

    relu relu_activation[9:0] (.data_in(dense2_res), .data_out(relu_res)); //ReLu activation
    
    assign layer_out = relu_res;               
    assign layer_done = dense2_done;

endmodule

module neural_network(
    input clk,
    input enable,
    input reset,
    input [7:0] img [0:783],
    output [7:0] digit_out,
    output NN_done
    );
    
    /* Average pooling layer */
    
    reg pool_enable;
    wire finished_pool;
    reg signed [15:0] pool [0:195];
    
    // Pixel value registers
    wire signed [7:0] pool_in1;
    wire signed [7:0] pool_in2;
    wire signed [7:0] pool_in3;
    wire signed [7:0] pool_in4;
    wire signed [7:0] pool_final;
    
    // Pixel address registers
    reg [15:0] pool_in1_addr;
    reg [15:0] pool_in2_addr;
    reg [15:0] pool_in3_addr;
    reg [15:0] pool_in4_addr;
    reg [15:0] pool_final_addr = 0;
    reg [15:0] pool_addr = 0;
    reg [15:0] pool_row = 0;
    
    // Initialize addresses
    initial
    begin
        pool_in1_addr <= 8'b0000_0000;
        pool_in2_addr <= 8'b0000_0001;
        pool_in3_addr <= 8'b0001_1100;
        pool_in4_addr <= 8'b0001_1101;
        pool_enable <= 1'b1;
    end
    
    avg_pooling AvgPooling(clk,pool_enable,pool_in1,pool_in2,pool_in3,pool_in4,pool_final,finished_pool);
    
    // Load pixel values
    assign pool_in1 = ((img[pool_in1_addr]));
    assign pool_in2 = ((img[pool_in2_addr]));
    assign pool_in3 = ((img[pool_in3_addr]));
    assign pool_in4 = ((img[pool_in4_addr]));
    
    always @(posedge clk) begin
    if(reset) begin
        pool_in1_addr <= 8'b0000_0000;
        pool_in2_addr <= 8'b0000_0001;
        pool_in3_addr <= 8'b0001_1100;
        pool_in4_addr <= 8'b0001_1101;
        pool_final_addr <= 0;
        pool_row <= 0;
        pool_addr <= 0;
        pool_enable <= 1'b1; 
    end
    else if(enable) begin
        if(finished_pool) begin // Average done
            pool[pool_final_addr] = pool_final;
            pool_addr <= pool_addr + 2; // Increment address
            pool_row <= pool_row + 2;
            if(pool_row == 28) begin // End of row, go down by 2 rows
                pool_addr <= pool_addr + pool_row;
                pool_row <= 0;
            end
            if(pool_in4_addr == 783) begin // Global averaging done
                pool_enable <= 0;
            end
            else if(pool_in4_addr != 783) begin // Update addresses
                pool_in1_addr <= pool_addr;
                pool_in2_addr <= pool_addr + 1;
                pool_in3_addr <= pool_addr + 28;
                pool_in4_addr <= pool_addr + 29;
                pool_final_addr <= pool_final_addr + 1;
            end
        end
    end       
    end
    
    /* Hidden layer */
    
    reg dense1_enable;
    wire finished_dense1;
    reg signed [15:0] dense1_res [0:31];
    initial dense1_enable <= 0;
    
    dense_layer1 layer2 (.clk(clk), .enable(dense1_enable), .reset(reset), .pooled_img(pool), .layer_out(dense1_res), .layer_done(finished_dense1));

    always @(posedge clk) begin
        if(reset) begin
            dense1_enable <= 0;
        end
        else if(enable) begin
            if(pool_enable == 0 && finished_dense1 == 0) begin // Pooling done
                dense1_enable <= 1;
            end
            else dense1_enable <= 0; // Hidden layer done
        end
    end
    
    /* Output layer */
    
    reg dense2_enable;
    wire finished_dense2;
    reg signed [15:0] dense2_res [0:9];
    initial dense2_enable <= 0;

    dense_layer2 layer3 (.clk(clk), .enable(dense2_enable), .reset(reset), .in_data(dense1_res), .layer_out(dense2_res), .layer_done(finished_dense2));

    always @(posedge clk) begin
        if(reset) begin
            dense2_enable <= 0;
        end
        else if(enable) begin
            if(pool_enable == 0 && finished_dense1 == 1 && finished_dense2 == 0) begin // Previous layers done
                dense2_enable <= 1;
            end
            else dense2_enable <= 0; // Output layer done
        end
    end
    
    /* Handwritten digit selection layer */
    
    reg max_enable;
    wire digit_recog_done;
    reg [7:0] digit;
    
    initial max_enable <= 0;
    
    select_max last_layer (.clk(clk), .enable(max_enable), .reset(reset), .in_data(dense2_res), .digit(digit), .layer_done(digit_recog_done));
        
    always @(posedge clk) begin
        if(reset) begin
            max_enable <= 0;
        end
        else if(enable) begin
            if(finished_dense2 == 1) begin // Output layer done
                max_enable <= 1;
            end
            else max_enable <= 0;
        end
    end
    
    assign digit_out = digit;
    assign NN_done = digit_recog_done;
    
endmodule


module neuron #(parameter IN_SIZE=196, WIDTH = 8)(
    input clk,
    input en,
    input reset,
    input signed [2*WIDTH-1:0] in_data[0:IN_SIZE-1],
    input signed[WIDTH-1:0] weight[0:IN_SIZE-1],
    input signed[WIDTH-1:0] bias,
    output signed[4*WIDTH-1:0] neuron_out,
    output neuron_done
    );
    
    integer addr = 0;
    reg done = 0;
    
    reg signed [4*WIDTH-1:0] product = 0;
    reg signed [4*WIDTH-1:0] out = 0;
    
    always @(posedge clk) begin
        if(reset) begin 
            done <= 0;
            addr <= 0;
        end
        else if(en) begin
            if(addr < IN_SIZE-1) begin
                product <= in_data[addr]*weight[addr]; //Calculate weighted input
                out <= out+product; //Sum each weighted input
               
            end
            if(addr == IN_SIZE-1) begin //Neuron output available
                done <= 1;
            end else begin
                addr <= addr + 1'b1;
                done <= 0;
            end
        end
    end
    
    assign neuron_out = out + bias; //Add bias
    assign neuron_done = done;
    
endmodule


module relu #(parameter WIDTH = 8)(
    input signed [4*WIDTH-1:0] data_in,
    output signed [2*WIDTH-1:0] data_out
    );
    
    wire signed [4*WIDTH-1:0] temp;
    
    assign temp = (data_in > 0)? data_in:0; //Take data_in if > 0, 0 else
    assign data_out = temp >> 8; //Rescale element and store into data_out
    
endmodule

module select_max # (parameter NEURON_NB=10, WIDTH=8)(
    input clk,
    input enable,
    input reset,
    input signed[2*WIDTH-1:0] in_data [0:NEURON_NB-1],
    output [WIDTH-1:0] digit,
    output layer_done
    );
    
    integer i = 0;
    reg signed [2*WIDTH-1:0] max = 0;
    reg signed [WIDTH-1:0] index = 0;
    reg done = 0;
    
    always @(posedge clk) begin
        if(reset) begin
            done <= 0;
            i <= 0;
            max <= 0;
            index <= 0;
        end
        else if(enable) begin
            if (in_data[i] >= max) begin //Update maximum and max index
                    max <= in_data[i]; 
                    index <= i;
                end
            if(i < 10) i <= i + 1;
                else done <= 1;
        end
    end
    
    assign digit = index;
    assign layer_done = done;
    
endmodule

