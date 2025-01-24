classdef Predictor
    %Predictor Create a predit object for the gluing methodology
    %   Given 2 regressors and one classifier build a general predictor
    %   that reconstruct x given z.

    properties
        classifier;
        mdl_a;
        mdl_b;
        sigma_a;
        sigma_b;
        mu_a;
        mu_b;
    end

    methods
        function obj = Predictor(models)
            %Predictor Construct an instance of Predictor
            %   svmModels is a classifier that returns 1 if the z point is
            %   located after a jump, 0 otherwise. 
            %   mdl_b is the regressor trained on the points before jumps.     
            %   mu_b and sigma_b are the associated normalization coefficients.            
            %   mdl_a is the regressor trained on the points after jumps.
            %   mu_a and sigma_a are the associated normalization coefficients.
              
            obj.classifier = models.randomForest;
            obj.mdl_b = models.mdl_b;
            obj.mdl_a = models.mdl_a;
            obj.sigma_a = models.sigma_a;
            obj.sigma_b = models.sigma_b;
            obj.mu_a = models.mu_a;
            obj.mu_b = models.mu_b;
        end

        function x_predicted = predict(obj, z)
            %predict Given z, reconstruct x
            %   Determine which model to use thanks to the classifier and
            %   apply it. 

            % Classify z points
            after_jumps_label = str2double(predict(obj.classifier, z));

            % Calculate results of both networks
            x_pred_before_jump = predict(obj.mdl_b, (z - obj.mu_b) ./ obj.sigma_b);
            x_pred_after_jump = predict(obj.mdl_a, (z - obj.mu_a) ./ obj.sigma_a);
            
            % Assign the correct result
            x_predicted = x_pred_before_jump;
            x_predicted(after_jumps_label == 1,:) = x_pred_after_jump(after_jumps_label == 1,:);
        end
    end
end